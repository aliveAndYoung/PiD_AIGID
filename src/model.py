import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_pid_resnet():
    # Initialize the model
    # Load pretrained model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Modify the final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    return model