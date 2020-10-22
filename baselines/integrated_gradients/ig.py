import numpy as np


def integrated_gradients(
    inputs,
    model,
    target_label_idx,
    get_gradients,
    baseline,
    device,
    steps=50,
    softmax=False,
):
    if baseline is None:
        baseline = 0 * inputs
    # scale inputs and compute gradients
    scaled_inputs = [
        baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)
    ]
    grads = get_gradients(
        scaled_inputs, model, target_label_idx, device, softmax=softmax
    )
    avg_grads = np.average(grads[:-1], axis=0)
    integrated_grad = (inputs - baseline) * avg_grads
    return integrated_grad
