import lightning.pytorch as pl
import torchvision
import torchmetrics
from torch import nn
import torch
import src.plot_utils as plot_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from itertools import chain
import numpy as np


class COASTMobileNetV2(pl.LightningModule):
    def __init__(self, **config):
        super().__init__()

        # Number of classes (default to 8 for transfer learning)
        self.num_classes = config.get("num_classes", 8)

        # early stopping config
        self.bad_epoch_count = 0
        self.max_bad_epochs = 15

        self.nesterov = config["nesterov"]
        # Learning rate scheduler configurations
        self.factor = config["factor"]
        self.patience = config["patience"]

        self.test_step_outputs = []


        # Optimizer configurations
        self.lr = config["lr"]
        self.momentum = config["momentum"]
        self.weight_decay = config["weight_decay"]

        #classifier layer configurations
        self.dropout = config["dropout"]
        self.n_layer1 = config["n_layer1"]
        self.n_layer2 = config["n_layer2"]
        self.batch_norm = config["batch_norm"]
        
        self.highest_val_acc = 0

        self.use_pretrained = config["use_pretrained"]
        self.fine_tune = config["fine_tune"]
        self.unfreeze_depth = config["unfreeze_depth"] * 3 # each layer has 3 parameters
        self.checkpoint = config["checkpoint"]

        # Load the MobileNetV2 model
        if self.use_pretrained:
            print("Using pretrained model with MobileNetV2_Weights.IMAGENET1K_V2")
            self.backbone = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2)
        else:
            print("Using model without pretrained weights")
            self.backbone = torchvision.models.mobilenet_v2(weights=None)

        # Get the number of input features for the FC layer
        input_features = self.backbone.classifier[1].in_features
        print(f"Number of input features: {input_features}")
        
        # Define classifier layers dynamically
        layers = [
            nn.Linear(input_features, self.n_layer1),
        ]

        if self.batch_norm:
            print("Using batch normalization for first hidden layer")
            layers.append(nn.BatchNorm1d(self.n_layer1))

        layers.extend([
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_layer1, self.n_layer2),
        ])

        if self.batch_norm:
            print("Using batch normalization for second hidden layer")
            layers.append(nn.BatchNorm1d(self.n_layer2))

        layers.extend([
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_layer2, self.num_classes),
        ])

        
        self.backbone.classifier = nn.Sequential(*layers)


        # **Freeze convolutional layers** (only train classifier)
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        # Print trainable parameters for debugging
        print("All layers in the network:")
        for name, param in self.backbone.named_parameters():
            print(f"  {name}, requires_grad={param.requires_grad}")

        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction="mean")  


        print(f"Fine-tuning: {self.fine_tune}")


        # Metrics
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.precision = torchmetrics.Precision(task="multiclass", num_classes=self.num_classes)
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=self.num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)

        # Load from checkpoint if provided
        if self.fine_tune:
            self.unfreeze_layers(self.unfreeze_depth)
            print(f"Loading model checkpoint from: {config.get('checkpoint')}")
            checkpoint = torch.load(config.get("checkpoint"), map_location=self.device)
            self.load_state_dict(checkpoint["state_dict"], strict=True)

        print("All layers in the network after unfreezing and loading checkpoint:")
        for name, param in self.backbone.named_parameters():
            print(f"  {name}, requires_grad={param.requires_grad}")

    def unfreeze_layers(self, unfreeze_depth):
        """
        Unfreeze the specified number of layers in the backbone features,
        starting from the last layers and moving backward.
        
        :param unfreeze_depth: Number of layers to unfreeze.
        """
        print(f"Unfreezing the last {unfreeze_depth} layers from features...")
        
        # Get all parameters from the backbone.features only.
        feature_params = list(self.backbone.features.named_parameters())
        
        # Reverse the order to start from the last feature layers
        for idx, (name, param) in enumerate(reversed(feature_params)):
            if idx < unfreeze_depth:
                param.requires_grad = True
                print(f"Unfreezing {name}")





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


    def forward(self, x, return_features=False):

        # ResNet-50 already normalizes input features internally
        # x_normalized = self.batch_norm(x)  # Normalize input features
        x = self.backbone(x)
        # x = self.pooling(x).squeeze()  # Apply pooling , already included in the ResNet-50 model
        # x = self.classifier(x)

        if return_features:
            return x[:, :-1]  # Extract second-to-last layer
        return x # Pass input features through model


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)


        self.log_dict(
            {
                'loss': loss,
                'acc': acc,
            },
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        val_loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        val_acc = self.accuracy(preds, y)

        self.log_dict( 
            {
                'val_loss': val_loss,
                'val_acc': val_acc,
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

        outputs = {'y_true': y.cpu().numpy(), 'y_pred': preds.cpu().numpy()}
        self.test_step_outputs.append(outputs)

        return test_loss


    def on_validation_epoch_end(self):
        """
        Check if training accuracy has exceeded validation accuracy for more than 'max_bad_epochs' epochs.
        If so, stop training early.
        """

        
        train_acc = self.trainer.callback_metrics.get("acc", None)
        val_acc = self.trainer.callback_metrics.get("val_acc", None)

        if self.current_epoch > 1:
            print(f"Plotting validation metrics for epoch {self.current_epoch}...")
            plot_utils.plot_metrics(self.logger.log_dir)

        if train_acc is not None and val_acc is not None and not self.fine_tune:
            if train_acc > val_acc:
                self.bad_epoch_count += 1
                print(f"⚠️ Training accuracy ({train_acc:.4f}) > Validation accuracy ({val_acc:.4f}) for {self.bad_epoch_count} epochs.")
            else:
                self.bad_epoch_count = 0  # Reset counter if validation accuracy catches up

            if self.bad_epoch_count >= self.max_bad_epochs:
                print("❌ Early stopping triggered: Training accuracy exceeded validation accuracy for too long.")
                self.trainer.should_stop = True  # Stop training
        


        if self.fine_tune and train_acc is not None and val_acc is not None:
            if self.highest_val_acc < val_acc:
                self.highest_val_acc = val_acc
                self.bad_epoch_count = 0
            if train_acc > .98 and val_acc < self.highest_val_acc:
                self.bad_epoch_count += 1
                print(f"⚠️ Training accuracy ({train_acc:.4f}) > Validation accuracy ({val_acc:.4f}) for {self.bad_epoch_count} epochs.")
            if self.bad_epoch_count >= self.max_bad_epochs:
                print("❌ Early stopping triggered: Training accuracy exceeded validation accuracy for too long.")
                self.trainer.should_stop = True


    def on_test_epoch_end(self):
        print("Plotting test metrics...")

        # Extract embeddings and labels for visualization
        sillhoute_embeddings = []
        all_embeddings = []
        all_labels = []
        y_true = np.concatenate([x['y_true'] for x in self.test_step_outputs])
        y_pred = np.concatenate([x['y_pred'] for x in self.test_step_outputs])

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

                embeddings = self.backbone(x).cpu().numpy()
                sillhoute_embeddings.append(self.forward(x, return_features=True).cpu().numpy())
                all_embeddings.append(embeddings)
                all_labels.append(y.cpu().numpy())  # Store labels as numpy arrays
                

                
        sillhoute_embeddings = np.vstack(sillhoute_embeddings)  # Ensure (n_samples, n_features) shape
        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.concatenate(all_labels).flatten()  # ✅ Ensure 1D array

        # Plot PCA and t-SNE visualizations
        plot_utils.plot_pca(all_embeddings, labels=all_labels, label_names=label_names, save_path=f"{self.logger.log_dir}/pca.png")
        plot_utils.plot_tsne(all_embeddings, labels=all_labels, label_names=label_names, save_path=f"{self.logger.log_dir}/tsne.png")
        plot_utils.plot_confusion_matrix(y_true, y_pred, classes=label_names,save_path=f"{self.logger.log_dir}/confusion.png")
        plot_utils.plot_silhouette(sillhoute_embeddings, labels=all_labels, label_names=label_names, save_path=f"{self.logger.log_dir}/silhouette.png")
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            params=filter(lambda p: p.requires_grad, self.parameters()),  # Get all unfrozen params
            lr=self.lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
            weight_decay=self.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            factor=self.factor,
            patience=self.patience
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }


class COASTvitb16(pl.LightningModule):
    def __init__(self, **config):
        super().__init__()

        # Number of classes (default to 8 for transfer learning)
        self.num_classes = config.get("num_classes", 8)

        # early stopping config
        self.bad_epoch_count = 0
        self.max_bad_epochs = 15
        self.highest_val_acc = 0

        # Optimizer configurations
        self.lr = config["lr"]
        self.momentum = config["momentum"]
        self.nesterov = config["nesterov"]
        self.weight_decay = config["weight_decay"]

        # Learning rate scheduler configurations
        self.factor = config["factor"]
        self.patience = config["patience"]

        #classifier layer configurations
        self.dropout = config["dropout"]
        self.n_layer1 = config["n_layer1"]
        self.n_layer2 = config["n_layer2"]
        self.batch_norm = config["batch_norm"]

        if self.batch_norm == 'true':
            self.batch_norm = True
        else:
            self.batch_norm = False

        # regarding fine-tuning
        self.use_pretrained = config["use_pretrained"]
        self.fine_tune = config["fine_tune"]
        self.checkpoint = config["checkpoint"]
        self.unfreeze_depth = config["unfreeze_depth"] 

        self.test_step_outputs = []

        # Load the model
        if self.use_pretrained:
            print("Using pretrained model with vit b 16")
            self.backbone = torchvision.models.vit_b_16(weights = 'DEFAULT')
        else:
            print("Using model without pretrained weights")
            self.backbone = torchvision.models.vit_b_16(weights = None)
       
       # Get the input feature size of ViT's classification head
        input_features = self.backbone.heads.head.in_features
        print(f"ViT feature dimension: {input_features}")


        layers = [
            nn.Linear(input_features, self.n_layer1),
        ]

        if self.batch_norm:
            print("Using batch normalization for first hidden layer")
            layers.append(nn.BatchNorm1d(self.n_layer1))
        
        layers.extend([
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_layer1, self.n_layer2),
        ])

        if self.batch_norm:
            print("Using batch normalization for second hidden layer")
            layers.append(nn.BatchNorm1d(self.n_layer2))
    
        layers.extend([
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_layer2, self.num_classes),
        ])

        self.backbone.heads.head = nn.Sequential(*layers)

        self.freeze_backbone()  # Freeze all layers in the backbone


        # Print all layers
        print("All layers in the network:")
        for name, param in self.backbone.named_parameters():
            print(f"  {name}, requires_grad={param.requires_grad}")

        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        # Metrics
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.precision = torchmetrics.Precision(task="multiclass", num_classes=self.num_classes)
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=self.num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)
        
        if self.fine_tune:
            print(f'self.fine_tune: {self.fine_tune} and self.checkpoint: {self.checkpoint}')
            self.unfreeze_layers(self.unfreeze_depth)
            self.load_checkpoint(self.checkpoint)

        print("All layers in the network after unfreezing:")
        for name, param in self.backbone.named_parameters():
            print(f"  {name}, requires_grad={param.requires_grad}")

    def freeze_backbone(self):
        """
        Freeze all layers in the backbone (feature extractor) while keeping the classifier trainable.
        """
        # Freeze all parameters in the backbone
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

        # Ensure classifier layers remain trainable
        for name, param in self.backbone.heads.head.named_parameters():
            param.requires_grad = True

        # Print frozen and trainable layers for debugging
        print("Trainable layers after freeze_backbone():")
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                print(f"  Backbone: {name}")
                
        print("Frozen layers after freeze_backbone():")
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                print(f"  Backbone: {name}")

        print("All layers in the network:")
        for name, param in self.backbone.named_parameters():
            print(f"  {name}, requires_grad={param.requires_grad}")


    def load_checkpoint(self, checkpoint_path):
        """
        Load a model checkpoint from a file.
        :param checkpoint_path: Path to the checkpoint file.
        """
        print(f"Loading model checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"], strict=True)  # `strict=False` allows new layers


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

    def unfreeze_layers(self, unfreeze_depth):
        """
        Unfreezes layers of the model starting from the last encoder layer
        and moving backward up to a specified depth.

        Parameters:
            unfreeze_depth (int): The number of layers to unfreeze from the last layer backward.
        """

        max_layers = 11 # 12 layers, 0 to 11
        if unfreeze_depth > max_layers:
            unfreeze_depth = max_layers
            print(f"Unfreezing all layers from encoder_layer_0 to encoder_layer_{max_layers}.")

        layers_to_unfreeze = []

        for i in range(unfreeze_depth+1):
            layers_to_unfreeze.append(max_layers - i + 1)   # 12 - 1 = 11, 12 - 2 = 10, ...

        for layer in layers_to_unfreeze:
            for name, param in self.backbone.named_parameters():
                # Only unfreeze parameters belonging to the target encoder layers
                if f"encoder.layers.encoder_layer_{layer}." in name:
                    param.requires_grad = True


    def forward(self, x, return_features=False):

        # ResNet-50 already normalizes input features internally
        # x_normalized = self.batch_norm(x)  # Normalize input features
        x = self.backbone(x)
        # x = self.pooling(x).squeeze()  # Apply pooling , already included in the ResNet-50 model
        # x = self.classifier(x)
        if return_features:
            return x[:, :-1]  # Extract second-to-last layer

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

        outputs = {'y_true': y.cpu().numpy(), 'y_pred': preds.cpu().numpy()}
        self.test_step_outputs.append(outputs)

        return test_loss


    def on_validation_epoch_end(self):
        train_acc = self.trainer.callback_metrics.get("acc", None)
        val_acc = self.trainer.callback_metrics.get("val_acc", None)

        if self.current_epoch > 1:
            print(f"Plotting validation metrics for epoch {self.current_epoch}...")
            plot_utils.plot_metrics(self.logger.log_dir)

        if train_acc is not None and val_acc is not None and not self.fine_tune:
            if train_acc > val_acc:
                self.bad_epoch_count += 1
                print(f"⚠️ Training accuracy ({train_acc:.4f}) > Validation accuracy ({val_acc:.4f}) for {self.bad_epoch_count} epochs.")
            else:
                self.bad_epoch_count = 0  # Reset counter if validation accuracy catches up

            if self.bad_epoch_count >= self.max_bad_epochs:
                print("❌ Early stopping triggered: Training accuracy exceeded validation accuracy for too long.")
                self.trainer.should_stop = True  # Stop training

        if self.fine_tune and train_acc is not None and val_acc is not None:
            if self.highest_val_acc < val_acc:
                self.highest_val_acc = val_acc
                self.bad_epoch_count = 0
            if train_acc > .98 and val_acc < self.highest_val_acc:
                self.bad_epoch_count += 1
                print(f"⚠️ Training accuracy ({train_acc:.4f}) > Validation accuracy ({val_acc:.4f}) for {self.bad_epoch_count} epochs.")
            if self.bad_epoch_count >= self.max_bad_epochs:
                print("❌ Early stopping triggered: Training accuracy exceeded validation accuracy for too long.")
                self.trainer.should_stop = True


    def on_test_epoch_end(self):
        print("Plotting test metrics...")

        # Extract embeddings and labels for visualization
        all_embeddings = []
        all_labels = []
        sillhoute_embeddings = []
        y_true = np.concatenate([x['y_true'] for x in self.test_step_outputs])
        y_pred = np.concatenate([x['y_pred'] for x in self.test_step_outputs])
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
                sillhoute_embeddings.append(self.forward(x, return_features=True).cpu().numpy())
                embeddings = self.backbone(x).cpu().numpy()
                all_embeddings.append(embeddings)
                all_labels.append(y.cpu().numpy())  # Store labels as numpy arrays


        sillhoute_embeddings = np.vstack(sillhoute_embeddings)  # Ensure (n_samples, n_features) shape
        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.concatenate(all_labels).flatten()  # ✅ Ensure 1D array

        # Plot PCA and t-SNE visualizations
        plot_utils.plot_pca(all_embeddings, labels=all_labels, label_names=label_names, save_path=f"{self.logger.log_dir}/pca.png")
        plot_utils.plot_tsne(all_embeddings, labels=all_labels, label_names=label_names, save_path=f"{self.logger.log_dir}/tsne.png")
        plot_utils.plot_confusion_matrix(y_true, y_pred, classes=label_names,save_path=f"{self.logger.log_dir}/confusion.png")
        plot_utils.plot_silhouette(sillhoute_embeddings, labels=all_labels, label_names=label_names, save_path=f"{self.logger.log_dir}/silhouette.png")


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            params=filter(lambda p: p.requires_grad, self.parameters()),  # Get all unfrozen params
            lr=self.lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
            weight_decay=self.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            factor=self.factor,
            patience=self.patience
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }
