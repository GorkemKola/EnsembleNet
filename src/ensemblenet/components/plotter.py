# imports
import os
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ensemblenet.entity.config import PlotConfig
from ensemblenet.utils import logger
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


class Plotter:
    def __init__(self, config: PlotConfig) -> None:
        self.config = config
        self.figsize = config.figsize
        plt.style.use('default')
        sns.set_style(config.sns_style)
        self.colors = sns.color_palette(config.color_palette, 10)

   
        
    def plot_all_metrics(self, save_path: str = None, show_plot: bool = True):
        """
        Plot all training metrics for multiple models.
        
        Args:
            model_names: List of model names
            dataframes: List of corresponding pandas DataFrames with training metrics
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
        """
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        fig.suptitle('Training Metrics Comparison Across Models', fontsize=16, fontweight='bold')
        
        # Define metrics to plot
        metrics = [
            ('train_loss', 'Training Loss', axes[0, 0]),
            ('valid_loss', 'Validation Loss', axes[0, 1]),
            ('accuracy', 'Accuracy (%)', axes[0, 2]),
            ('precision', 'Precision', axes[1, 0]),
            ('recall', 'Recall', axes[1, 1]),
            ('f1_score', 'F1 Score', axes[1, 2])
        ]
        
        # Plot each metric
        for metric, title, ax in metrics:
            self._plot_metric(metric, title, ax)
        
        # Add learning rate plot as an inset or separate info
        self._add_learning_rate_info(fig)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def _plot_metric(self,
                    metric: str, title: str, ax):
        """Plot a single metric for all models."""
        model_names = self.config.model_names
        dataframes = self.config.training_metrics
        
        for i, (name, df) in enumerate(zip(model_names, dataframes)):
            if metric in df.columns:
                epochs = df['epoch'] + 1  # Start from epoch 1
                values = df[metric]
                
                # Use different colors for each model
                color = self.colors[i % len(self.colors)]
                ax.plot(epochs, values, marker='o', linewidth=2, 
                       markersize=4, label=name, color=color, alpha=0.8)
                
                # Highlight best model epochs
                if 'is_best_model' in df.columns:
                    best_epochs = df[df['is_best_model'] == True]
                    if not best_epochs.empty:
                        best_x = best_epochs['epoch'] + 1
                        best_y = best_epochs[metric]
                        ax.scatter(best_x, best_y, color=color, s=100, 
                                 marker='*', edgecolors='black', linewidth=1,
                                 zorder=5)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Format y-axis based on metric type
        if metric == 'accuracy':
            ax.set_ylim(0, 100)
        elif metric in ['precision', 'recall', 'f1_score']:
            ax.set_ylim(0, 1)

    def _add_learning_rate_info(self, fig):
        """Add learning rate information to the plot."""
        model_names = self.config.model_names
        dataframes = self.config.training_metrics
        
        lr_info = []
        for name, df in zip(model_names, dataframes):
            if 'learning_rate' in df.columns:
                initial_lr = df['learning_rate'].iloc[0]
                final_lr = df['learning_rate'].iloc[-1]
                lr_changes = len(df['learning_rate'].unique())
                lr_info.append(f"{name}: {initial_lr:.6f} → {final_lr:.6f} ({lr_changes} changes)")
        
        if lr_info:
            fig.text(0.02, 0.02, "Learning Rate Info:\n" + "\n".join(lr_info), 
                    fontsize=8, verticalalignment='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
            
        
    def plot_loss_comparison(self,
                           save_path: str = None, show_plot: bool = True):
        """
        Create a focused comparison of training and validation losses.
        """
        model_names = self.config.model_names
        dataframes = self.config.training_metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Loss Comparison Across Models', fontsize=14, fontweight='bold')
        
        for i, (name, df) in enumerate(zip(model_names, dataframes)):
            epochs = df['epoch'] + 1
            color = self.colors[i % len(self.colors)]
            
            # Training loss
            ax1.plot(epochs, df['train_loss'], marker='o', linewidth=2, 
                    markersize=3, label=name, color=color, alpha=0.8)
            
            # Validation loss
            ax2.plot(epochs, df['valid_loss'], marker='s', linewidth=2, 
                    markersize=3, label=name, color=color, alpha=0.8)
        
        ax1.set_title('Training Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Validation Loss', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    def plot_accuracy_comparison(self,
                                save_path: str = None, show_plot: bool = True):
        """
        Create a focused comparison of accuracy metrics.
        """
        model_names = self.config.model_names
        dataframes = self.config.training_metrics
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle('Accuracy Comparison Across Models', fontsize=14, fontweight='bold')
        for i, (name, df) in enumerate(zip(model_names, dataframes)):
            epochs = df['epoch'] + 1
            color = self.colors[i % len(self.colors)]
            
            ax.plot(epochs, df['accuracy'], marker='o', linewidth=2, 
                    markersize=3, label=name, color=color, alpha=0.8)
            
            # Highlight best model epochs
            if 'is_best_model' in df.columns:
                best_epochs = df[df['is_best_model'] == True]
                if not best_epochs.empty:
                    best_x = best_epochs['epoch'] + 1
                    best_y = best_epochs['accuracy']
                    ax.scatter(best_x, best_y, color=color, s=100, 
                             marker='*', edgecolors='black', linewidth=1,
                             zorder=5)
        ax.set_title('Accuracy', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        return fig
    def plot_precision_recall_f1(self,
                                 save_path: str = None, show_plot: bool = True):
        """
        Create a focused comparison of precision, recall, and F1 score.
        """
        model_names = self.config.model_names
        dataframes = self.config.training_metrics
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Precision, Recall, and F1 Score Comparison Across Models', fontsize=14, fontweight='bold')
        metrics = ['precision', 'recall', 'f1_score']
        titles = ['Precision', 'Recall', 'F1 Score']
        for i, (metric, title, ax) in enumerate(zip(metrics, titles, axes)):
            for j, (name, df) in enumerate(zip(model_names, dataframes)):
                epochs = df['epoch'] + 1
                color = self.colors[j % len(self.colors)]
                
                ax.plot(epochs, df[metric], marker='o', linewidth=2, 
                        markersize=3, label=name, color=color, alpha=0.8)
                
                # Highlight best model epochs
                if 'is_best_model' in df.columns:
                    best_epochs = df[df['is_best_model'] == True]
                    if not best_epochs.empty:
                        best_x = best_epochs['epoch'] + 1
                        best_y = best_epochs[metric]
                        ax.scatter(best_x, best_y, color=color, s=100, 
                                 marker='*', edgecolors='black', linewidth=1,
                                 zorder=5)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        return fig
    def plot_learning_rate(self,
                           save_path: str = None, show_plot: bool = True):
        """
        Create a focused plot of learning rate changes.
        """

        model_names = self.config.model_names
        dataframes = self.config.training_metrics
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle('Learning Rate Changes Across Models', fontsize=14, fontweight='bold')
        for i, (name, df) in enumerate(zip(model_names, dataframes)):
            if 'learning_rate' in df.columns:
                epochs = df['epoch'] + 1
                color = self.colors[i % len(self.colors)]
                
                ax.plot(epochs, df['learning_rate'], marker='o', linewidth=2, 
                        markersize=3, label=name, color=color, alpha=0.8)
                # Highlight best model epochs
                if 'is_best_model' in df.columns:
                    best_epochs = df[df['is_best_model'] == True]
                    if not best_epochs.empty:
                        best_x = best_epochs['epoch'] + 1
                        best_y = best_epochs['learning_rate']
                        ax.scatter(best_x, best_y, color=color, s=100, 
                                 marker='*', edgecolors='black', linewidth=1,
                                 zorder=5)
        ax.set_title('Learning Rate', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        return fig
    
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
    
        
    def plot_test_results(self, save_path: str = None, columns: List[str] = None, show_plot: bool = True):
        """
        Plot each test metric in a separate bar chart.
        Args:
            save_path: Optional directory to save the plots (each metric will be saved as a separate file)
            columns: List of columns to plot (default is all except 'model')
            show_plot: Whether to display the plot
        """
        results_df = self.config.test_results.copy()
        if columns is None:
            columns = results_df.columns.tolist()
            columns.remove('model')
        for metric in columns:
            fig, ax = plt.subplots(figsize=self.config.figsize)
            colors = sns.color_palette(self.config.color_palette, n_colors=len(results_df))
            results_df.set_index('model')[metric].plot(kind='bar', ax=ax, color=colors, alpha=0.8)
            ax.set_title(f'{metric.capitalize()} Comparison Across Models', fontsize=16, fontweight='bold')
            ax.set_ylabel(metric.capitalize())
            ax.set_xlabel('Model')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            if save_path:
                # Save each metric plot as a separate file
                plt.savefig(os.path.join(save_path, f"test_{metric}_barchart.png"), dpi=300, bbox_inches='tight')
            if show_plot:
                plt.show()
            plt.close(fig)
        return

    def get_number_of_params(self, model_path) -> Dict[str, any]:
        """
        Get the number of parameters in a model.
        
        Args:
            model: The model instance
        
        Returns:
            A dictionary with the number of parameters
        """
        import torch
        model = torch.load(
            model_path, 
            map_location=torch.device('cpu'),
            weights_only=False
        )
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"num_params": num_params}
    
    def extract_model_params(self, save_path: str = None, show_plot: bool = True):
        """
        Plot the number of parameters for each model.
        
        Args:
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
        """
        num_params = [self.get_number_of_params(model_name).get("num_params") for model_name in self.config.model_paths]
        
        # save to csv
        params_df = pd.DataFrame({
            'model': self.config.model_names,
            'num_params': num_params
        })
        params_csv_path = self.config.root_dir / "model_params.csv"
        params_df.to_csv(params_csv_path, index=False)
        logger.info(f"Saved model parameters to {params_csv_path}")