import lightning.pytorch as pl
import torchvision
import torchmetrics
from torch import nn
import torch
import src.plot_utils as plot_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from itertools import chain
import numpy as np

class COASTALResNet50(pl.LightningModule):
  def __init__(self, **config):
        super().__init__() # Initialize the parent LightningModule class

        # Number of classes
        self.num_classes  = config["num_classes"]

        # Optimizer configurations
        self.lr           = config["lr"]
        self.momentum     = config["momentum"]
        self.nesterov     = config["nesterov"]
        self.weight_decay = config["weight_decay"]
        self.frozen_layers= config["frozen_layers"]

        # Learning rate scheduler configurations
        self.factor   = config["factor"]
        self.patience = config["patience"]

        self.use_pretrained = config["use_pretrained"]
        self.fine_tune = config["fine_tune"]

        if self.use_pretrained:
            # model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1, num_classes=self.num_classes)
            print("Using pretrained model with ResNet50_Weights.IMAGENET1K_V1")
            self.backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            print("Using model without pretrained weights")
            self.backbone = torchvision.models.resnet50(weights=None)

        # Get the number of input features for the fc before removing it
        input_features = self.backbone.fc.in_features

        # Remove the default FC layer
        self.backbone.fc = nn.Identity()

        # Add the Classifier
        # self.classifier = nn.Sequential(
          # nn.Linear(input_features, 1024),  # First hidden layer
          # nn.BatchNorm1d(1024),
          # nn.ReLU(),
          # nn.Dropout(0.5),  # Dropout for regularization

          # nn.Linear(1024, 512),  # Second hidden layer
          # nn.BatchNorm1d(512),
          # nn.ReLU(),
          # nn.Dropout(0.5),

          # nn.Linear(512, 256),  # Third hidden layer
          # nn.BatchNorm1d(256),
          # nn.ReLU(),
          # nn.Dropout(0.5),

          # nn.Linear(256, self.num_classes)  # Output layer
        # )

        self.classifier = nn.Sequential(nn.Linear(input_features, self.num_classes))

        # Adjust the architecture for 224x224 images
        self.backbone.conv1 = nn.Conv2d(
            in_channels=self.backbone.conv1.in_channels,
            out_channels=self.backbone.conv1.out_channels,
            kernel_size=(7,7), # for 224x244 images
            stride=(2,2),      # for 224x244 images
            padding=self.backbone.conv1.padding,
            bias=False
        )

        # Print the model's trainable parameters
        print("Trainable parameters:")
        for name, param in chain(self.backbone.named_parameters(), self.classifier.named_parameters()):
          print(f"{name}: {param.requires_grad}")

        # Apply freezing logic
        if not self.fine_tune:
            self.freeze_backbone()  # Freeze all layers except the classifier

        # Apply specific layer freezing logic
        # Requires that the model can be fine-tuned
        if self.fine_tune and self.frozen_layers:
            self.freeze_layers(self.frozen_layers)

        # Batch normalization layer, already implemented in ResNet-50
        # self.batch_norm = nn.BatchNorm2d(num_features=model.conv1.in_channels)

        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        # Metrics to track
        self.accuracy  = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.precision = torchmetrics.Precision(task="multiclass", num_classes=self.num_classes)
        self.recall    = torchmetrics.Recall(task="multiclass", num_classes=self.num_classes)
        self.f1_score  = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)


  def freeze_backbone(self):
      """
      Freeze all layers in the backbone (feature extractor) while keeping the classifier trainable.
      """
      # Freeze all parameters in the backbone
      for name, param in self.backbone.named_parameters():
          param.requires_grad = False

      # Print frozen and trainable layers for debugging
      print("Trainable layers after freeze_backbone():")
      for name, param in self.backbone.named_parameters():
          if param.requires_grad:
              print(f"  Backbone: {name}")
      for name, param in self.classifier.named_parameters():
          if param.requires_grad:
              print(f"  Classifier: {name}")

      print("Frozen layers after freeze_backbone():")
      for name, param in self.backbone.named_parameters():
          if not param.requires_grad:
              print(f"  Backbone: {name}")


  def freeze_layers(self, frozen_layers):
      """
      Freeze the specified layers in the backbone.
      :param frozen_layers: List of layer names to freeze (e.g., ['layer1', 'layer2']).
      """
      # Freeze the parameters in the specified layers
      for name, param in self.backbone.named_parameters():
          if any(layer in name for layer in frozen_layers):
              param.requires_grad = False

      # Print frozen and trainable layers for debugging
      print("Trainable layers after freeze_layers():")
      for name, param in self.backbone.named_parameters():
          if param.requires_grad:
              print(f"  Backbone: {name}")
      for name, param in self.classifier.named_parameters():
          if param.requires_grad:
              print(f"  Classifier: {name}")

      print("Frozen layers after freeze_layers():")
      for name, param in self.backbone.named_parameters():
          if not param.requires_grad:
              print(f"  Backbone: {name}")


  def forward(self, x):

        # ResNet-50 already normalizes input features internally
        # x_normalized = self.batch_norm(x)  # Normalize input features
        x = self.backbone(x)
        # x = self.pooling(x).squeeze()  # Apply pooling , already included in the ResNet-50 model
        x = self.classifier(x)
        return x # Pass input features through model


  def training_step(self, batch, batch_idx):
        x, y = batch # Input features (x) and labels (y)

        # Perform a forward pass through the model to get predictions
        logits = self.forward(x)

        # Calculate the loss
        loss = self.criterion(logits, y)

        # Compute the predicted class labels
        preds = torch.argmax(logits, dim=1)

        # Log metrics
        self.log_dict(
            {
                'loss': loss,
                'acc': self.accuracy(preds, y),
            },
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        return loss # return loss for backpropagation


  def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)

        val_loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.log_dict(
            {
                'val_loss'     : val_loss,
                'val_acc'      : self.accuracy(preds, y),
                ##### Uncomment for additional metrics #####
                # 'val_precision': self.precision(preds, y),
                # 'val_recall'   : self.recall(preds, y),
                # 'val_F-1 score': self.f1_score(preds, y),
            },
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        return val_loss


  def test_step(self, batch, batch_idx):
        print("On test step...")
        x, y = batch

        logits = self.forward(x)

        test_loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.log_dict(
            {
                'test_loss'     : test_loss,
                'test_acc'      : self.accuracy(preds, y),
                ##### Uncomment for additional metrics #####
                # 'val_precision': self.precision(preds, y),
                # 'val_recall'   : self.recall(preds, y),
                # 'val_F-1 score': self.f1_score(preds, y),
            },
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        return test_loss


  def on_validation_epoch_end(self):
      if self.current_epoch > 1:
          plot_utils.plot_metrics(self.logger.log_dir)


  def on_test_epoch_end(self):
      print("Plotting test metrics...")

      # Extract embeddings and labels for visualization
      all_embeddings = []
      all_labels = []

      dataloader = self.trainer.datamodule.test_dataloader()
      self.eval()  # Ensure model is in eval mode
      device = next(self.parameters()).device  # Get model device (CPU or GPU)

      # Fetch dataset & extract label mapping
      dataset = dataloader.dataset  # Access the ImageFolder dataset
      label_names = {v: k for k, v in dataset.class_to_idx.items()}  # Maps indices to category names

      with torch.no_grad():
          for batch in dataloader:
              x, y = batch
              x = x.to(device)  # Move input to the same device as model
              y = y.to(device)

              embeddings = self.backbone(x).cpu().numpy()  # Move output to CPU before converting to numpy
              all_embeddings.append(embeddings)
              all_labels.append(y.cpu().numpy())  # Store labels as numpy arrays

      all_embeddings = np.vstack(all_embeddings)
      all_labels = np.concatenate(all_labels).flatten()  # ✅ Ensure 1D array

      # Plot PCA and t-SNE visualizations
      plot_utils.plot_pca(all_embeddings, labels=all_labels, label_names=label_names, save_path=f"{self.logger.log_dir}/pca.png")
      plot_utils.plot_tsne(all_embeddings, labels=all_labels, label_names=label_names, save_path=f"{self.logger.log_dir}/tsne.png")


  def configure_optimizers(self):
      optimizer = torch.optim.SGD(
          params       = chain(self.backbone.parameters(), self.classifier.parameters()),
          lr           = self.lr,
          momentum     = self.momentum,
          nesterov     = self.nesterov,
          weight_decay = self.weight_decay,
      )

      scheduler = ReduceLROnPlateau(
          optimizer = optimizer,
          factor    = self.factor,
          patience  = self.patience
      )

      return {
          "optimizer": optimizer,
          "lr_scheduler": {
              "scheduler" : scheduler,
              "interval"  : "epoch",
              "frequency" : 1,
              "monitor"   : "val_loss",
          },
      }