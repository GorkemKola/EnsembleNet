import os
from ensemblenet.constants import *
from ensemblenet.utils import read_yaml, create_directories
from ensemblenet.entity.config import (
    DataIngestionConfig, 
    TrainingConfig,
    TestConfig,
    PlotConfig
)
import pandas as pd
from ensemblenet.utils import logger

class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH,
    ) -> None:
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([
            Path(self.config.artifacts_root),

            # Data Ingestion Directories
            Path(self.config.data_ingestion.root_dir),
            Path(self.config.data_ingestion.unzip_dir),
            Path(self.config.data_ingestion.train_dir),
            Path(self.config.data_ingestion.test_dir),
            Path(self.config.data_ingestion.val_dir),
            
            # Training Directories
            Path(self.config.training.root_dir),
            Path(self.config.training.tensorboard_log_dir),

            # Test Directories
            Path(self.config.test.root_dir),
            Path(self.config.test.results_dir),

            # Plot Directories
            Path(self.config.plot.root_dir),
        ])
        
        allowed_models = [
            "resnet", 
            "inception_v3", 
            "vit", 
            "squeezenet", 
            "shufflenet", 
            "mobilenet", 
            "mnasnet",
            "ensemblenet",
            "ensemblenet0",
            "ensemblenet1",
            "ensemblenet2"
        ]
        if self.params.MODEL_NAME not in allowed_models:
             raise ValueError(f"Invalid MODEL_NAME '{self.params.MODEL_NAME}' in params.yaml. Choose from {allowed_models}")
        self.model_name = self.params.MODEL_NAME
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_kaggle_dataset_id=config.source_kaggle_dataset_id,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
            train_dir=Path(config.train_dir),
            test_dir=Path(config.test_dir),
            val_dir=Path(config.val_dir),
            cleanup_unzip_dir_after_split=config.cleanup_unzip_dir_after_split # Read from yaml
        )

        return data_ingestion_config
    
    def get_training_config(self) -> TrainingConfig:
        config_train = self.config.training
        config_data = self.config.data_ingestion
        config_ensemble = self.config.prepare_ensemble_model
        params = self.params

        # --- Get Data Paths ---
        train_data_path = Path(config_data.train_dir)
        valid_data_path = Path(config_data.val_dir)
        test_data_path = Path(config_data.test_dir)
        # --- End ---

        # --- Get Classes and Pretrained Flag ---
        num_classes = params.CLASSES
        # Determine if pretrained weights should be used based on param
        use_pretrained = params.PRETRAINED_WEIGHTS is not None and params.PRETRAINED_WEIGHTS.upper() != 'NONE'
        logger.info(f"Number of classes set to: {num_classes}")
        logger.info(f"Use pretrained weights: {use_pretrained} (Based on PRETRAINED_WEIGHTS: {params.PRETRAINED_WEIGHTS})")
        # --- End ---

        # --- Get Ensemble Model Path ---
        ensemble_model_path = Path(config_ensemble.ensemble_model_path)
        if self.model_name == "attention_ensemble" and not ensemble_model_path.exists():
             logger.warning(f"Selected MODEL_NAME is 'attention_ensemble', but the model file "
                            f"at {ensemble_model_path} does not exist!")

        # --- Construct model-specific save paths using templates ---
        last_model_path = Path(config_train.last_model_path_template.format(model_name=self.model_name))
        best_model_path = Path(config_train.best_model_path_template.format(model_name=self.model_name))
        create_directories([last_model_path.parent, best_model_path.parent]) # Ensure parent dirs exist
        # --- End ---

        # --- Log Image Size Warning (as before) ---
        expected_size = None
        if self.model_name == "inception_v3": expected_size = (299, 299)
        elif self.model_name == "vit": expected_size = (224, 224)
        current_size = tuple(params.IMAGE_SIZE[:2])
        if expected_size and current_size != expected_size:
            logger.warning(f"Model '{self.model_name}' typically expects input size {expected_size}, "
                           f"but IMAGE_SIZE is set to {current_size}.")
        # --- End Log ---

        find_lr_flag = getattr(params, 'FIND_LR', False) # Default to False if not in params
        logger.info(f"Run Learning Rate Finder: {find_lr_flag}")
        # --- Construct LR finder plot path ---
        lr_plot_filename = f"lr_finder_plot_{self.model_name}.png"
        lr_finder_plot_path = Path(config_train.root_dir) / lr_plot_filename
        create_directories([lr_finder_plot_path.parent]) # Ensure parent dir exists

        training_config = TrainingConfig(
            root_dir=Path(config_train.root_dir),
            last_model_path=last_model_path,
            best_model_path=best_model_path,
            train_data_path=train_data_path,
            valid_data_path=valid_data_path,
            test_data_path=test_data_path,
            attention_ensemble_model_path=ensemble_model_path,
            model_name=self.model_name,
            num_classes=num_classes,         
            use_pretrained=use_pretrained,   
            lr_finder_plot_path=lr_finder_plot_path,    
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_early_stopping_patience=params.EARLY_STOPPING_PATIENCE,
            params_learning_rate=params.LEARNING_RATE,
            params_random_state=params.RANDOM_STATE,
            params_find_lr=find_lr_flag,                    
        )
        
        return training_config
    
    def get_test_config(self) -> TestConfig:
        # Define model names based on your training artifacts
        model_names = sorted(self.config.test.model_names)
        best_model_paths = [self.config.test.best_model_path_template.format(model_name=name) for name in model_names]
        hyperparams_paths = [self.config.test.hyperparams_path_template.format(model_name=name) for name in model_names]
        hyperparams = [pd.read_csv(path) for path in hyperparams_paths]
        training_metric_paths = [self.config.test.training_metrics_path_template.format(model_name=name) for name in model_names]
        training_metrics = [pd.read_csv(path) for path in training_metric_paths]
        test_config = TestConfig(
            root_dir=Path(self.config.test.root_dir),
            models_dir=Path(self.config.test.models_dir),
            results_dir=Path(self.config.test.results_dir),
            test_dir=Path(self.config.data_ingestion.test_dir),
            best_model_paths=best_model_paths,
            hyperparams_paths=hyperparams_paths,
            model_names=model_names,
            params=self.params,
            hyperparams=hyperparams,
            training_metrics=training_metrics,
            batch_size=self.params.BATCH_SIZE
        )
        
        return test_config
    
    def get_plot_config(self) -> PlotConfig:
        model_names = sorted(self.config.test.model_names)
        model_paths = [self.config.test.best_model_path_template.format(model_name=name) for name in model_names]
        training_metric_paths = [self.config.plot.training_metrics_path_template.format(model_name=name) for name in model_names]
        training_metrics = [pd.read_csv(path) for path in training_metric_paths]
        test_results_path = self.config.plot.test_results_path
        test_results = pd.read_csv(test_results_path) if Path(test_results_path).exists() else pd.DataFrame()
        plot_config = PlotConfig(
            root_dir=Path(self.config.plot.root_dir),
            model_names=model_names,
            params=self.params,
            training_metrics=training_metrics,
            test_results=test_results,
            model_paths=[Path(path) for path in model_paths],
            figsize=self.params.FIGSIZE,
            color_palette=self.params.COLOR_PALETTE,
            sns_style=self.params.SNS_STYLE,
        )
        
        return plot_config