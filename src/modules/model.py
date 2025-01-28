import lightning.pytorch as pl
import torchvision
import torchmetrics
from torch import nn
import torch
import src.plot_utils as plot_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from itertools import chain

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
            # model = torchvision.models.resnet50(weights=None, num_classes=self.num_classes)
            print("Using model without pretrained weights")
            self.backbone = torchvision.models.resnet50(weights=None)

        # Get the number of input features for the fc before removing it
        input_features = self.backbone.fc.in_features

        # Remove the default FC layer
        self.backbone.fc = nn.Identity()

        # Generate Custom Fully Connected Layers
        fc_layers = []

        # Add the Final Classifier
        fc_layers.append(nn.Linear(input_features, self.num_classes))
        self.classifier = nn.Sequential(*fc_layers)

        # Adjust the architecture for 224x224 images
        self.backbone.conv1 = nn.Conv2d(
            in_channels=self.backbone.conv1.in_channels,
            out_channels=self.backbone.conv1.out_channels,
            kernel_size=(7,7), # for 224x244 images
            stride=(2,2),      # for 224x244 images
            padding=self.backbone.conv1.padding,
            bias=False
        )

        # Combine the backbone and classifier
        # model = nn.Sequential(self.backbone, self.classifier)

        # Since images are 299x299, max pooling could actually help reduce computation and improve feature hierarchy.
        # model.maxpool = nn.Identity() # Remove the first pooling layer

        # Assign the model to an instance variable
        # self.resnet50 = model


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
        # x = self.pooling(x).squeeze()  # Apply pooling
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
      plot_utils.plot_test_metrics(self.logger.log_dir)


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