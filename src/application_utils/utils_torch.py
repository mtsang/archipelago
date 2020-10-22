import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import torch.optim as optim


class ModelWrapperTorch:
    def __init__(self, model, device, input_type="image"):
        self.device = device
        self.model = model.to(device)
        self.input_type = input_type

    def __call__(self, X):
        if self.input_type == "text":
            X = torch.LongTensor(X).to(self.device)
            preds = self.model(X)[0].data.cpu().numpy()
        else:
            X = torch.FloatTensor(X).to(self.device)
            if self.input_type == "image":
                X = X.permute(0, 3, 1, 2)
            preds = self.model(X).data.cpu().numpy()
        return preds
