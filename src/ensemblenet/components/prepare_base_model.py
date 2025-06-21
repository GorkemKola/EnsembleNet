import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchsummary import summary
from pathlib import Path

class PrepareBaseModel:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_base_model(self):
        self.model = resnet50(weights=self.config.params_weights)
        if not self.config.params_include_top:
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.save_model(path=self.config.base_model_path, model=self.model)

    def _prepare_full_model(self, model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False

        elif (freeze_till is not None) and (freeze_till > 0):
            for param in list(model.parameters())[:-freeze_till]:
                param.requires_grad = False

        if self.config.params_include_top:
            model.fc = nn.Linear(model.fc.in_features, classes)
        else:
            model.add_module('flatten', nn.Flatten())
            model.add_module('fc', nn.Linear(2048, classes))

        model = model.to(self.device)

        return model
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: nn.Module):
        torch.save(model, path)

    def load_model(self, path: Path):
        self.model = torch.load(path)
        self.model.to(self.device)

    def summary(self):
        if self.full_model is None:
            raise ValueError("Model hasn't been prepared yet. Call update_base_model() first.")
        
        # Adjusting input_size format for torchsummary
        # Assuming self.config.IMAGE_SIZE is in the format [height, width, channels]
        input_size = (self.config.params_image_size[2], self.config.params_image_size[0], self.config.params_image_size[1])
        
        summary(self.full_model, input_size=input_size, device=str(self.device))
    