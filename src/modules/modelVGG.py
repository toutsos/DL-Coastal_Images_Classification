import lightning.pytorch as pl
import torchvision
import torchmetrics
from torch import nn
import torch
import src.plot_utils as plot_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt

class COASTAL_VGG(pl.LightningModule):
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
        self.load_ckpt    = config["load_ckpt"]
        self.ckpt_path   = config["ckpt_path"]

        # Learning rate scheduler configurations
        self.factor   = config["factor"]
        self.patience = config["patience"]

        self.use_pretrained = config["use_pretrained"]
        self.fine_tune      = config["fine_tune"]
        self.dropout        = config["dropout"]

        self.test_results   = []



        if self.use_pretrained and not self.load_ckpt:
            print("Using pretrained model with VGG16_Weights.IMAGENET1K_V1")
            self.backbone = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        else:
            print("Using model without pretrained weights")
            self.backbone = torchvision.models.vgg16(weights=None)

        # Get the number of input features for the fc before removing it
        input_features = self.backbone.classifier[0].in_features


        # Remove the default FC layer
        self.backbone.classifier = nn.Identity()

        # self.classifier = nn.Sequential(
        #     nn.Linear(input_features, 1024),  
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout),  # Higher dropout to prevent overfitting   
        #     nn.Linear(1024, self.num_classes)  # Final classification layer  
        # )

        self.classifier = nn.Sequential(
            nn.Linear(input_features, 1024),  
            nn.ReLU(),
            nn.Dropout(self.dropout),  # Higher dropout to prevent overfitting
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout), 
            nn.Linear(512, self.num_classes)  # Final classification layer  
        )

        # Define penultimate layers (exclude the final classification layer
        self.penultimate = nn.Sequential(
            *list(self.backbone.classifier.children())[:-1]  # Exclude the last linear layer from the classifier
        )



        # Print the model's trainable parameters
        print("Trainable parameters:")
        for name, param in chain(self.backbone.named_parameters(), self.classifier.named_parameters()):
          print(f"{name}: {param.requires_grad}")

        # Apply freezing logic

        if self.load_ckpt:

            checkpoint = torch.load(self.ckpt_path, map_location=self.device)
            c = checkpoint['state_dict']
            for name, tensor in c.items():
                print(f"{name}: {tensor.shape}")
            self.load_state_dict(checkpoint['state_dict'], strict=True)


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
      # Freeze the parameters in the specified layer
      frozen_layers = ['features.' + str(layer) for layer in frozen_layers]
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

  def unfreeze_backbone(self):
      """
      Freeze all layers in the backbone (feature extractor) while keeping the classifier trainable.
      """
      # Freeze all parameters in the backbone
      for name, param in self.backbone.named_parameters():
          param.requires_grad = True

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


  def forward(self, x, project_embeddings=None):

        # ResNet-50 already normalizes input features internally
        # x_normalized = self.batch_norm(x)  # Normalize input features
        x = self.backbone(x)

        if project_embeddings == 'backbone':
            return x
        elif project_embeddings == 'penultimate':
            x = self.penultimate(x)
            return x
        else:

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

        x, y = batch

        bb_embeddings = self.forward(x, 'backbone')
        pen_embeddings = self.forward(x, 'penultimate')
        logits = self.classifier(bb_embeddings)
        #logits = self.forward(x)

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


        for i in range(x.shape[0]):  # Append test batch embeddings and other info to test results
            self.test_results.append({
                'bb_embedding': bb_embeddings[i].cpu().numpy(),
                'pen_embedding': pen_embeddings[i].cpu().numpy(),
                'label': y[i].item(),
                'prediction': preds[i].item(),
                'test_loss': test_loss.item()  # Scalar loss (same for all instances)
            })

        return test_loss


  def on_validation_epoch_end(self):
      if self.current_epoch > 1:
          plot_utils.plot_metrics(self.logger.log_dir)



  def on_test_epoch_end(self):
      print("Plotting test metrics...")

      # Extract embeddings and labels for visualization
      bb_embeddings = []
      all_labels = []
      all_preds = []
      pen_embeddings = []

      # Fetch dataset & extract label mapping
      dataloader = self.trainer.datamodule.test_dataloader()
      dataset = dataloader.dataset  # Access the ImageFolder dataset
      label_names = {v: k for k, v in dataset.class_to_idx.items()}  # Maps indices to category names

      # Loop through outputs from each batch
      for output in self.test_results:
          bb_embeddings.append(output['bb_embedding'])
          pen_embeddings.append(output['pen_embedding'])
          all_labels.append(output['label'])
          all_preds.append(output['prediction'])

      bb_embeddings = np.vstack(bb_embeddings)
      pen_embeddings = np.vstack(pen_embeddings)
      all_labels = np.array(all_labels)
      all_preds = np.array(all_preds)

      # Plot PCA and t-SNE visualizations, along with confusion matrix
      plot_utils.plot_pca(bb_embeddings, labels=all_labels, label_names=label_names, save_path=f"{self.logger.log_dir}/pca.png")
      plot_utils.plot_tsne(bb_embeddings, labels=all_labels, label_names=label_names, save_path=f"{self.logger.log_dir}/tsne.png")
      plot_utils.plot_confusion_matrix(preds=all_preds, labels=all_labels, label_names=label_names, save_path=f"{self.logger.log_dir}/confusion_matrix.png")
      plot_utils.plot_silhouette_analysis(embeddings=pen_embeddings, save_path=f"{self.logger.log_dir}/sil2.png", range_n_clusters=[8])

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