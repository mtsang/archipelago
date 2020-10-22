import numpy as np


def get_efficient_mask_indices(inst, baseline, input):
    invert = np.sum(1 * inst) >= len(inst) // 2
    if invert:
        context = input.copy()
        insertion_target = baseline
        mask_indices = np.argwhere(inst == False).flatten()
    else:
        context = baseline.copy()
        insertion_target = input
        mask_indices = np.argwhere(inst == True).flatten()
    return mask_indices, context, insertion_target
