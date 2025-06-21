import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ensemblenet.entity.config import TestConfig
from ensemblenet.utils import logger

class Test:
    def __init__(self, config: TestConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")


    def _load_model(self, model_path: Path) -> nn.Module:
        print(model_path)
        model = torch.load(
            model_path,
            map_location=torch.device('cpu'),
            weights_only=False
        )
        return model.to(self.device)
    
    def load_models(self) -> List[nn.Module]:
        models = {}
        for i, model_path in enumerate(self.config.best_model_paths):
            model_name = self.config.model_names[i]
            model = self._load_model(model_path)
            models[model_name] = model
            logger.info(f"Loaded model from {model_path}")
            print(f"Loaded model from {model_path}")
        return models
        
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
    
    def load_test_dataset(self, data_dir: Path, image_size=(224, 224), batch_size=64, sample_limit=None):
        """
        Loads the test dataset from the specified directory.
        """
        mean, std = self.compute_dataset_mean_std(data_dir, image_size, batch_size, sample_limit)
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return loader
    
    def evaluate_model(self, model_name, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluates the model on the given dataloader.
        Returns a dictionary with accuracy, precision, recall, f1-score, and AUC.
        """
        
        model.eval()
        all_preds = []
        all_labels = []
        # compute evaluation time
        logger.info(f"Evaluating model: {model.__class__.__name__}")
        print(f"Evaluating model: {model.__class__.__name__}")
        start_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        if start_time:
            start_time.record()
        criterion = nn.CrossEntropyLoss()
        losses = []
        
        # add top5 accuracy
        top5_correct = 0
        top5_total = 0
        if start_time:
            start_time.record()
        if self.device.type == 'cuda':
            model = model.to(self.device)
            criterion = criterion.to(self.device)
        else:
            model = model.cpu()
            criterion = criterion.cpu()
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
                # Store predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                # Compute top-5 accuracy
                _, top5_preds = torch.topk(outputs, 5, dim=1)
                top5_correct += (top5_preds == labels.view(-1, 1)).sum().item()
                top5_total += labels.size(0)
        # Compute top-5 accuracy
        top5_accuracy = top5_correct / top5_total if top5_total > 0 else 0
        logger.info(f"Top-5 Accuracy: {top5_accuracy:.4f}")
        
        if start_time:
            end_time = torch.cuda.Event(enable_timing=True)
            end_time.record()
            torch.cuda.synchronize()
            eval_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
        
        loss = np.mean(losses)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        # Compute confusion matrix and classification report
            
        results_path = self.config.results_dir / "results.csv"
        if not results_path.exists():
            pd.DataFrame(columns=["model", "loss", "accuracy", "top 5 accuracy", "precision", "recall", "f1"]).to_csv(results_path, index=False)
        results_df = pd.read_csv(results_path)
        new_row = {
            "model": model_name,
            "loss": loss,
            "accuracy": accuracy,
            "top 5 accuracy": top5_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "time": eval_time if start_time else None
        }

        new_row = pd.DataFrame([new_row])
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved evaluation results to {results_path}")

    def evaluate_all_models(self):
        """
        Loads all models and evaluates them
        """
        models = self.load_models()
        test_loader = self.load_test_dataset(self.config.test_dir, 
                                             image_size=(224, 224), 
                                             batch_size=self.config.batch_size, )
        
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            self.evaluate_model(model_name, model, test_loader)