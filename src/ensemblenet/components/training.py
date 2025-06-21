import traceback
import math
import numpy as np
import pandas as pd
import csv
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ignite.engine import Engine
from ignite.handlers import FastaiLRFinder
import os

from ensemblenet.config.configuration import TrainingConfig
from ensemblenet.utils import logger
from ensemblenet.components.ensemblenet import EnsembleNet

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha (float or Tensor): Weighting factor (default: 1.0)
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): 'mean', 'sum', or 'none' (default: 'mean')
        ignore_index (int): Index to ignore (default: -100)
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs (Tensor): Model predictions (logits) of shape (N, C)
            targets (Tensor): Ground truth labels of shape (N,)
        
        Returns:
            Tensor: Focal loss
        """
        ce_loss = self.ce_loss(outputs, targets)
        p_t = torch.exp(-ce_loss)  # Convert back to probabilities
        
        # Handle ignore_index
        if self.ignore_index >= 0:
            ignore_mask = targets == self.ignore_index
            p_t = p_t.masked_fill(ignore_mask, 1.0)
        
        # Calculate focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        if isinstance(self.alpha, (float, int)):
            alpha_weight = self.alpha
        else:
            alpha_weight = self.alpha.gather(0, targets)
        
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        
        # Initialize CSV logging
        self.setup_csv_logging()
        
    def setup_csv_logging(self):
        """Setup CSV files for logging metrics and hyperparameters"""
        # Create logging directory if it doesn't exist
        log_dir = Path(self.config.root_dir) / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for unique file names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define CSV file paths
        self.metrics_csv_path = log_dir / f"training_metrics_{self.config.model_name}.csv"
        self.hyperparams_csv_path = log_dir / f"hyperparameters_{self.config.model_name}.csv"
        self.lr_finder_csv_path = log_dir / f"lr_finder_results_{self.config.model_name}.csv"
        
        # Initialize metrics CSV with headers
        metrics_headers = [
            'epoch', 'train_loss', 'valid_loss', 'accuracy', 'precision', 
            'recall', 'f1_score', 'learning_rate', 'is_best_model', 
            'early_stopping_counter', 'timestamp'
        ]
        
        with open(self.metrics_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(metrics_headers)
        
        logger.info(f"Metrics CSV initialized: {self.metrics_csv_path}")
        
        # Save hyperparameters immediately
        self.save_hyperparameters()
        
    def save_hyperparameters(self):
        """Save all hyperparameters and configuration to CSV"""
        hyperparams_data = []
        
        # Get all config attributes
        config_dict = vars(self.config)
        
        # Add system information
        hyperparams_data.extend([
            ('timestamp', datetime.now().isoformat()),
            ('device', str(self.device)),
            ('torch_version', torch.__version__),
            ('cuda_available', torch.cuda.is_available()),
            ('cuda_version', torch.version.cuda if torch.cuda.is_available() else 'N/A'),
        ])
        
        # Add configuration parameters
        for key, value in config_dict.items():
            hyperparams_data.append((key, str(value)))
        
        # Add derived parameters
        if hasattr(self, 'train_loader') and self.train_loader:
            hyperparams_data.extend([
                ('total_train_samples', len(self.train_loader.dataset)),
                ('train_batches_per_epoch', len(self.train_loader)),
            ])
        
        if hasattr(self, 'valid_loader') and self.valid_loader:
            hyperparams_data.extend([
                ('total_valid_samples', len(self.valid_loader.dataset)),
                ('valid_batches_per_epoch', len(self.valid_loader)),
            ])
            
        if hasattr(self, 'test_loader') and self.test_loader:
            hyperparams_data.extend([
                ('total_test_samples', len(self.test_loader.dataset)),
                ('test_batches_per_epoch', len(self.test_loader)),
            ])
        
        # Save to CSV
        df_hyperparams = pd.DataFrame(hyperparams_data, columns=['parameter', 'value'])
        df_hyperparams.to_csv(self.hyperparams_csv_path, index=False)
        
        logger.info(f"Hyperparameters saved to: {self.hyperparams_csv_path}")
        
    def log_metrics_to_csv(self, epoch, train_loss, valid_loss, accuracy, precision, 
                          recall, f1, learning_rate, is_best_model, early_stopping_counter):
        """Log training metrics to CSV"""
        timestamp = datetime.now().isoformat()
        
        metrics_row = [
            epoch + 1,  # 1-indexed for readability
            train_loss,
            valid_loss,
            accuracy,
            precision,
            recall,
            f1,
            learning_rate,
            is_best_model,
            early_stopping_counter,
            timestamp
        ]
        
        with open(self.metrics_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(metrics_row)
            
    def save_lr_finder_results(self, lr_finder, suggested_lr=None):
        """Save LR finder results to CSV, including suggested learning rate"""
        try:
            results = lr_finder.get_results()
            if results and 'lr' in results and 'loss' in results:
                df_lr = pd.DataFrame({
                    'learning_rate': results['lr'],
                    'loss': results['loss']
                })
                df_lr.to_csv(self.lr_finder_csv_path, index=False)
                logger.info(f"LR Finder results saved to: {self.lr_finder_csv_path}")
                
                # Save suggested learning rate to a separate file
                if suggested_lr is not None:
                    # Handle both Path objects and strings
                    if hasattr(self.lr_finder_csv_path, 'parent'):
                        # Path object
                        lr_suggestion_path = self.lr_finder_csv_path.parent / (self.lr_finder_csv_path.stem + '_suggested_lr.txt')
                    else:
                        # String path
                        lr_suggestion_path = str(self.lr_finder_csv_path).replace('.csv', '_suggested_lr.txt')
                    
                    with open(lr_suggestion_path, 'w') as f:
                        f.write(str(suggested_lr))
                    logger.info(f"Suggested learning rate ({suggested_lr:.2e}) saved to: {lr_suggestion_path}")
            else:
                logger.warning("No LR Finder results to save")
        except Exception as e:
            logger.error(f"Error saving LR Finder results: {e}")

    def load_suggested_lr(self):
        """Load previously saved suggested learning rate"""
        try:
            # Handle both Path objects and strings
            if hasattr(self.lr_finder_csv_path, 'parent'):
                # Path object
                lr_suggestion_path = self.lr_finder_csv_path.parent / (self.lr_finder_csv_path.stem + '_suggested_lr.txt')
            else:
                # String path
                lr_suggestion_path = str(self.lr_finder_csv_path).replace('.csv', '_suggested_lr.txt')
            
            if os.path.exists(lr_suggestion_path):
                with open(lr_suggestion_path, 'r') as f:
                    suggested_lr = float(f.read().strip())
                logger.info(f"Loaded previously suggested learning rate: {suggested_lr:.2e}")
                return suggested_lr
            else:
                logger.info("No previously saved suggested learning rate found")
                return None
        except Exception as e:
            logger.error(f"Error loading suggested learning rate: {e}")
            return None

    def find_optimal_lr_ignite(self, start_lr=1e-4, end_lr=1e-1, num_iter=100, step_mode="exp", force_recalculate=False):
        """
        Runs the LR Range Test using ignite.handlers.FastaiLRFinder.
        Returns a suggested learning rate and saves results to CSV.
        
        Args:
            end_lr: Maximum learning rate to test
            num_iter: Number of iterations
            step_mode: Step mode for learning rate schedule
            force_recalculate: If True, recalculates even if saved results exist
        """
        if not self.model or not self.train_loader:
            logger.error("Model and Train Loader must be initialized before running LR Finder.")
            return None

        # Check if we have a previously saved suggested learning rate
        if not force_recalculate:
            saved_lr = self.load_suggested_lr()
            if saved_lr is not None:
                logger.info(f"Using previously calculated suggested learning rate: {saved_lr:.2e}")
                logger.info("Set force_recalculate=True to recalculate from scratch")
                return saved_lr

        logger.info(f"--- Running Learning Rate Finder (Ignite) ---")
        logger.info(f"Using end_lr={end_lr}, num_iter={num_iter}, step_mode='{step_mode}'")

        self.model.train()
        criterion = nn.CrossEntropyLoss()
        temp_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-7)

        def update_fn(engine, batch):
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            temp_optimizer.zero_grad()
            outputs = self.model(inputs)
            if self.config.model_name == "attention_ensemble":
                logits = outputs
                loss = F.cross_entropy(logits, labels)
            elif isinstance(outputs, tuple) and hasattr(outputs, 'logits') and hasattr(outputs, 'aux_logits') and self.config.model_name == "inception_v3":
                loss = criterion(outputs.logits, labels)
            elif isinstance(outputs, tuple):
                loss = criterion(outputs[0], labels)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            temp_optimizer.step()
            return loss.item()

        finder_engine = Engine(update_fn)
        lr_finder = FastaiLRFinder()
        to_save = {"model": self.model, "optimizer": temp_optimizer}

        suggested_lr_final = None
        try:
            with lr_finder.attach(finder_engine, to_save=to_save, start_lr=start_lr, end_lr=end_lr) as trainer_with_lr_finder:
                trainer_with_lr_finder.run(self.train_loader, max_epochs=1)

            suggested_lr = lr_finder.lr_suggestion()

            logger.info("Plotting LR Finder results...")
            fig = lr_finder.plot()
            if fig:
                plot_path = self.config.lr_finder_plot_path
                logger.info(f"LR Finder plot saved to: {plot_path}")
            else:
                logger.warning("Could not generate LR Finder plot.")

            if suggested_lr is None:
                logger.warning("Ignite LR Finder could not suggest a learning rate. Using fallback calculation.")
                results = lr_finder.get_results()
                if results and 'lr' in results and 'loss' in results and len(results['loss']) > 0:
                    min_loss_idx = results['loss'].index(min(results['loss']))
                    min_loss_lr = results['lr'][min_loss_idx]
                    suggested_lr_final = min_loss_lr / 10.0
                    logger.info(f"LR at Minimum Loss: {min_loss_lr:.2e}")
                    logger.info(f"Fallback suggestion (Min Loss / 10): {suggested_lr_final:.2e}")
                else:
                    logger.warning("Could not retrieve results for fallback calculation.")
                    suggested_lr_final = None
            else:
                logger.info(f"Ignite LR Finder suggested learning rate: {suggested_lr:.2e}")
                suggested_lr_final = suggested_lr

            # Apply bounds checking
            if suggested_lr_final is not None:
                if suggested_lr_final > 0.1:
                    logger.warning(f"Suggested LR {suggested_lr_final:.2e} seems high. Clamping to 0.1. Please check the plot!")
                    suggested_lr_final = 0.1
                elif suggested_lr_final < 1e-7:
                    logger.warning(f"Suggested LR {suggested_lr_final:.2e} seems too low. Clamping to 1e-7. Please check the plot!")
                    suggested_lr_final = 1e-7

            # Save results including the suggested learning rate
            self.save_lr_finder_results(lr_finder, suggested_lr_final)

        except Exception as e:
            logger.error(f"Error during Ignite LR Finder execution: {e}")
            import traceback
            traceback.print_exc()
            suggested_lr_final = None

        logger.info("--- Learning Rate Finder (Ignite) Finished ---")
        return suggested_lr_final
    
    def build_model(self):
        """Builds/loads the specified model, using config flags."""
        logger.info(f"Building model: {self.config.model_name}")
        num_classes = self.config.num_classes
        use_pretrained = self.config.use_pretrained

        if self.config.model_name == "resnet":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None
            model = models.resnet50(weights=weights)
            log_msg = f"Loading ResNet50 with {'ImageNet pretrained' if use_pretrained else 'random'} weights."
            logger.info(log_msg)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            logger.info(f"Adapted ResNet FC layer for {num_classes} classes.")

        elif self.config.model_name == "inception_v3":
            weights = models.Inception_V3_Weights.IMAGENET1K_V1 if use_pretrained else None
            model = models.inception_v3(weights=weights, aux_logits=True, transform_input=True)
            log_msg = f"Loading Inception V3 with {'ImageNet pretrained' if use_pretrained else 'random'} weights."
            logger.info(log_msg)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            logger.info(f"Adapted Inception V3 main FC layer for {num_classes} classes.")
            if hasattr(model, 'AuxLogits') and model.AuxLogits is not None:
                num_ftrs_aux = model.AuxLogits.fc.in_features
                model.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes)
                logger.info(f"Adapted Inception V3 auxiliary FC layer for {num_classes} classes.")
            else:
                logger.info("Inception V3 AuxLogits not present or adapted.")

        elif self.config.model_name == "vit":
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if use_pretrained else None
            model = models.vit_b_16(weights=weights)
            log_msg = f"Loading ViT-B/16 with {'ImageNet pretrained' if use_pretrained else 'random'} weights."
            logger.info(log_msg)
            num_ftrs = model.heads.head.in_features
            model.heads.head = nn.Linear(num_ftrs, num_classes)
            logger.info(f"Adapted ViT classification head for {num_classes} classes.")

        elif self.config.model_name == "attention_ensemble":
            model_path = self.config.attention_ensemble_model_path
            logger.info(f"Loading Attention Ensemble model AND weights from: {model_path}")
            if not model_path.exists():
                raise FileNotFoundError(f"Attention ensemble model file not found at: {model_path}")
            try:
                model = torch.load(
                    model_path,
                    map_location=torch.device('cpu'),
                    weights_only=False
                )
                logger.info(f"Successfully loaded model structure and weights from {model_path}")
                final_layer_name = 'fc'
                if hasattr(model, final_layer_name):
                    final_layer = getattr(model, final_layer_name)
                    if isinstance(final_layer, nn.Linear):
                        num_ftrs = final_layer.in_features
                        if final_layer.out_features != num_classes:
                            logger.warning(f"Adapting final layer '{final_layer_name}' of loaded attention model. Output: {final_layer.out_features} -> {num_classes}")
                            setattr(model, final_layer_name, nn.Linear(num_ftrs, num_classes))
                        else:
                            logger.info(f"Final layer '{final_layer_name}' already matches class count ({num_classes}).")
                    else:
                        logger.warning(f"Attribute '{final_layer_name}' is not nn.Linear. Manual adaptation might be needed.")
                else:
                    logger.warning(f"Could not find final layer '{final_layer_name}'. Assuming loaded model output is correct.")
            except Exception as e:
                logger.error(f"Failed to load or adapt attention ensemble model from {model_path}: {e}")
                raise
        elif self.config.model_name == "ensemblenet":
            model = EnsembleNet(num_classes, embed_dim=768)

        elif self.config.model_name == "ensemblenet0":
            model = EnsembleNet(num_classes, embed_dim=768)
            weights = [
                "artifacts/training/model_mobilenet_best.pth",
                "artifacts/training/model_mnasnet_best.pth",
                "artifacts/training/model_squeezenet_best.pth"
            ]
                        
            model.load_backbone_weights(weights, strict=False)
            model.freeze_backbones()
        elif self.config.model_name == "ensemblenet1":
            model = torch.load(
                "artifacts/training/model_ensemblenet0_best.pth",
                map_location=torch.device('cpu'),
                weights_only=False
            )
            model.unfreeze_backbones()
            
        elif self.config.model_name == "ensemblenet2":
            model = EnsembleNet(num_classes, embed_dim=768)
            weights = [
                "artifacts/training/model_mobilenet_best.pth",
                "artifacts/training/model_mnasnet_best.pth",
                "artifacts/training/model_squeezenet_best.pth"
            ]
                        
            model.load_backbone_weights(weights, strict=False)
            model.unfreeze_backbones()

        elif self.config.model_name == "squeezenet":
            model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights)
            log_msg = f"Loading ViT-B/16 with {'ImageNet pretrained' if use_pretrained else 'random'} weights."
            logger.info(log_msg)
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Conv2d(512, num_classes, kernel_size=(1, 1)),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            )
        elif self.config.model_name == "shufflenet":
            model = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights)
            in_features = model._stage_out_channels[-1]
            model.fc = nn.Linear(in_features, num_classes)
            
        elif self.config.model_name == "mobilenet":
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights)
            lastconv_output_channels = 576
            last_channel = 1024
            dropout = 0.2
            model.classifier = nn.Sequential(
                nn.Linear(lastconv_output_channels, last_channel),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(last_channel, num_classes),
            )
        elif self.config.model_name == "attention_ensemble_better":
            pass
        elif self.config.model_name == "mnasnet":
            model = models.mnasnet0_5(weights=models.MNASNet0_5_Weights)
            dropout = 0.2
            model.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(1280, num_classes))
        else:
            raise ValueError(f"Unsupported model name: {self.config.model_name}")

        self.model = model.to(self.device)
        logger.info(f"Model '{self.config.model_name}' built/loaded and moved to {self.device}.")
        
        # Update hyperparameters CSV with model info after building
        self.save_hyperparameters()

    def compute_dataset_mean_std(self, data_dir, image_size=(224, 224), batch_size=64, sample_limit=None):
        """
        Computes mean and std for all images in a directory (ImageFolder structure).
        Uses cached .npy arrays if available, otherwise computes and saves them.
        """
        mean_path = Path(data_dir) / "mean.npy"
        std_path = Path(data_dir) / "std.npy"

        if mean_path.exists() and std_path.exists():
            logger.info(f"Loading cached mean/std from {mean_path} and {std_path}")
            mean = np.load(mean_path)
            std = np.load(std_path)
            return mean.tolist(), std.tolist()

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        n_images = 0
        mean = 0.
        std = 0.
        for i, (imgs, _) in enumerate(loader):
            if sample_limit and n_images >= sample_limit:
                break
            imgs = imgs.view(imgs.size(0), imgs.size(1), -1)
            mean += imgs.mean(2).sum(0)
            std += imgs.std(2).sum(0)
            n_images += imgs.size(0)
            if sample_limit and n_images >= sample_limit:
                break

        mean /= n_images
        std /= n_images

        np.save(mean_path, mean.cpu().numpy() if hasattr(mean, "cpu") else mean)
        np.save(std_path, std.cpu().numpy() if hasattr(std, "cpu") else std)
        logger.info(f"Saved computed mean/std to {mean_path} and {std_path}")

        return mean.tolist(), std.tolist()

    def prepare_data_loaders(self):
        """Prepares DataLoaders using pre-split train, validation, and test directories."""
        img_height, img_width, _ = self.config.params_image_size
        target_size = (img_height, img_width)
        mean, std = self.compute_dataset_mean_std(self.config.train_data_path)

        basic_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        if self.config.params_augmentation:
            train_transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            logger.info("Using data augmentation for training.")
        else:
            train_transform = basic_transform
            logger.info("Not using data augmentation for training.")

        logger.info(f"Loading training data from: {self.config.train_data_path}")
        train_dataset = datasets.ImageFolder(
            root=self.config.train_data_path,
            transform=train_transform
        )

        logger.info(f"Loading validation data from: {self.config.valid_data_path}")
        valid_dataset = datasets.ImageFolder(
            root=self.config.valid_data_path,
            transform=basic_transform
        )

        logger.info(f"Loading test data from: {self.config.test_data_path}")
        test_dataset = datasets.ImageFolder(
            root=self.config.test_data_path,
            transform=basic_transform
        )

        if not (train_dataset.classes == valid_dataset.classes == test_dataset.classes):
            logger.warning("Class inconsistency detected between train/validation/test splits!")
        logger.info(f"Found {len(train_dataset.classes)} classes.")
        if len(train_dataset.classes) != self.config.num_classes:
            logger.warning(f"Number of classes found in data ({len(train_dataset.classes)}) "
                          f"does not match params.yaml ({self.config.num_classes}). Check data paths and params.")

        num_workers = min(os.cpu_count() // 2, 8) if os.cpu_count() else 4
        logger.info(f"Using {num_workers} workers for DataLoaders.")

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        self.valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        logger.info(f"DataLoaders prepared:")
        logger.info(f"  Training samples: {len(train_dataset)}, Batches: {len(self.train_loader)}")
        logger.info(f"  Validation samples: {len(valid_dataset)}, Batches: {len(self.valid_loader)}")
        logger.info(f"  Test samples: {len(test_dataset)}, Batches: {len(self.test_loader)}")
        
        # Update hyperparameters with dataset info
        self.save_hyperparameters()

    @staticmethod
    def save_model(path: Path, model: nn.Module):
        logger.info(f"Saving model to: {path}")
        torch.save(model, path)

    def train(self):
        """Enhanced training method with comprehensive CSV logging and Cosine Annealing LR"""
        if not self.model or not self.train_loader or not self.valid_loader:
            logger.error("Model or DataLoaders not initialized. Call build_model() and prepare_data_loaders() first.")
            return

        logger.info(f"Starting training for model: {self.config.model_name}...")
        logger.info(f"Metrics will be logged to: {self.metrics_csv_path}")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.params_learning_rate)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=1-1/math.e,patience=self.config.params_early_stopping_patience // 4)
        
        logger.info(f"Using OneCycleLR scheduler with T_max={self.config.params_epochs}, eta_min=1e-6")

        best_valid_loss = float('inf')
        early_stopping_counter = 0

        for epoch in range(self.config.params_epochs):
            # Training Phase
            self.model.train()
            train_loss = 0.0
            train_pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.params_epochs} [Train]', leave=False)
            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                if self.config.model_name == "attention_ensemble":
                    logits = outputs
                    loss = F.cross_entropy(logits, labels)
                    display_loss = loss.item()
                elif isinstance(outputs, models.inception.InceptionOutputs) and self.model.training:
                    loss_main = criterion(outputs.logits, labels)
                    loss_aux = criterion(outputs.aux_logits, labels)
                    loss = loss_main + 0.4 * loss_aux
                    display_loss = loss.item()
                else:
                    outputs_logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    loss = criterion(outputs_logits, labels)
                    display_loss = loss.item()
                loss.backward()
                optimizer.step()

                train_loss += display_loss * inputs.size(0)
                train_pbar.set_postfix({'loss': f'{display_loss:.4f}', 'lr': f"{optimizer.param_groups[0]['lr']:.1e}"})
            avg_train_loss = train_loss / len(self.train_loader.dataset)

            # Validation Phase
            self.model.eval()
            valid_loss = 0.0
            all_labels = []
            all_predictions = []
            valid_pbar = tqdm(self.valid_loader, desc=f'Epoch {epoch+1}/{self.config.params_epochs} [Valid]', leave=False)
            with torch.no_grad():
                for inputs, labels in valid_pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    outputs_logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    loss = criterion(outputs_logits, labels)
                    valid_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs_logits.data, 1)
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
                    valid_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            avg_valid_loss = valid_loss / len(self.valid_loader.dataset)
            scheduler.step(avg_valid_loss)
            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_predictions) * 100
            precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
            current_lr = optimizer.param_groups[0]['lr']

            logger.info(f'Epoch {epoch+1} Summary -> Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, '
                       f'Valid Acc: {accuracy:.2f}%, Prec: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, LR: {current_lr:.2e}')

            # Determine if this is the best model
            is_best = avg_valid_loss < best_valid_loss
            
            # Log metrics to CSV
            self.log_metrics_to_csv(
                epoch=epoch,
                train_loss=avg_train_loss,
                valid_loss=avg_valid_loss,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                learning_rate=current_lr,
                is_best_model=is_best,
                early_stopping_counter=early_stopping_counter
            )
            
            # Save models
            self.save_model(path=self.config.last_model_path, model=self.model)
            
            if is_best:
                best_valid_loss = avg_valid_loss
                early_stopping_counter = 0
                self.save_model(path=self.config.best_model_path, model=self.model)
                logger.info(f'>>> Best model saved (Epoch {epoch+1}) with Valid Loss: {avg_valid_loss:.4f}')
            else:
                early_stopping_counter += 1
                logger.info(f'Validation loss did not improve. Counter: {early_stopping_counter}/{self.config.params_early_stopping_patience}')
                if early_stopping_counter >= self.config.params_early_stopping_patience:
                    logger.warning('--- Early stopping triggered ---')
                    break

        logger.info(f'Training finished for {self.config.model_name}.')
        logger.info(f'Best validation loss achieved: {best_valid_loss:.4f}')
        logger.info(f'Best model saved at: {self.config.best_model_path}')
        logger.info(f'Last model saved at: {self.config.last_model_path}')
        logger.info(f'Training metrics saved to: {self.metrics_csv_path}')
        logger.info(f'Hyperparameters saved to: {self.hyperparams_csv_path}')
        
        # Create a final summary CSV
        self.create_training_summary()
        
    def create_training_summary(self):
        """Create a summary CSV with final training results"""
        try:
            # Read the metrics CSV to get final results
            df_metrics = pd.read_csv(self.metrics_csv_path)
            if not df_metrics.empty:
                best_epoch_row = df_metrics.loc[df_metrics['is_best_model'] == True]
                final_epoch_row = df_metrics.iloc[-1]
                
                summary_data = {
                    'model_name': [self.config.model_name],
                    'total_epochs': [len(df_metrics)],
                    'best_epoch': [best_epoch_row['epoch'].iloc[0] if not best_epoch_row.empty else 'N/A'],
                    'best_valid_loss': [best_epoch_row['valid_loss'].iloc[0] if not best_epoch_row.empty else 'N/A'],
                    'best_accuracy': [best_epoch_row['accuracy'].iloc[0] if not best_epoch_row.empty else 'N/A'],
                    'final_train_loss': [final_epoch_row['train_loss']],
                    'final_valid_loss': [final_epoch_row['valid_loss']],
                    'final_accuracy': [final_epoch_row['accuracy']],
                    'final_f1_score': [final_epoch_row['f1_score']],
                    'early_stopped': [final_epoch_row['early_stopping_counter'] >= self.config.params_early_stopping_patience],
                    'training_completed': [datetime.now().isoformat()]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_path = Path(self.config.root_dir) / "logs" / f"training_summary_{self.config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                summary_df.to_csv(summary_path, index=False)
                
                logger.info(f"Training summary saved to: {summary_path}")
        except Exception as e:
            logger.error(f"Error creating training summary: {e}")