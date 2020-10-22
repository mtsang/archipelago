import torch
import torch.nn.functional as F

# import cv2
import numpy as np


def get_gradients(scaled_inputs, model, target_label_idx, device, softmax=False):
    grads = []
    for i, input in enumerate(scaled_inputs):
        torch_input = torch.FloatTensor(input).unsqueeze(0).to(device)
        torch_input.requires_grad = True
        pred = model(torch_input)
        if softmax:
            pred = F.softmax(pred, dim=1)
        output = pred[:, target_label_idx]
        model.zero_grad()
        output.backward()
        grad = torch_input.grad.detach().cpu().numpy()
        grads.append(grad)
    grads = np.concatenate(grads)
    return grads
