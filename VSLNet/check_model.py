import torch

from model.VSLNet import VSLNet
from utils.config import get_config

config = get_config()

model = VSLNet(config)
print("Model loaded")
