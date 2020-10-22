import torch
import numpy as np


class BertWrapperTorch:
    def __init__(self, model, device, merge_logits=False):
        self.model = model.to(device)
        self.device = device
        self.merge_logits = merge_logits

    def get_predictions(self, batch_ids):
        batch_ids = torch.LongTensor(batch_ids).to(self.device)
        batch_conf = self.model(batch_ids, None, None)
        if isinstance(batch_conf, tuple):
            return batch_conf[0].data.cpu()
        else:
            return batch_conf.data.cpu()
        return batch_conf

    def __call__(self, batch_ids):
        batch_predictions = self.get_predictions(batch_ids)
        if self.merge_logits:
            batch_predictions2 = (
                (batch_predictions[:, 1] - batch_predictions[:, 0]).unsqueeze(1).numpy()
            )
            return batch_predictions2
        else:
            return batch_predictions.numpy()
