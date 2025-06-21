from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import pandas as pd

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_kaggle_dataset_id: str
    local_data_file: Path
    unzip_dir: Path
    train_dir: Path
    test_dir: Path
    val_dir: Path
    cleanup_unzip_dir_after_split: bool

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    last_model_path: Path
    best_model_path: Path

    train_data_path: Path
    valid_data_path: Path
    test_data_path: Path

    attention_ensemble_model_path: Path
    model_name: str
    num_classes: int
    use_pretrained: bool
    lr_finder_plot_path: Path
    params_epochs: int
    params_batch_size: int
    params_augmentation: bool
    params_image_size: list
    params_early_stopping_patience: int
    params_learning_rate: float
    params_random_state: int
    params_find_lr: bool


@dataclass(frozen=True)
class TestConfig:
    root_dir: Path
    models_dir: Path
    results_dir: Path
    test_dir: Path
    best_model_paths: List[Path]
    hyperparams_paths: List[Path]
    model_names: List[str]
    params: Dict[str, any]
    hyperparams: List[pd.DataFrame]
    training_metrics: List[pd.DataFrame]
    batch_size: int
    
@dataclass(frozen=True)
class PlotConfig:
    root_dir: Path
    model_names: List[str]
    params: Dict[str, any]
    training_metrics: List[pd.DataFrame]
    figsize: List[int]
    color_palette: str
    sns_style: str
    test_results: pd.DataFrame
    model_paths: List[Path]