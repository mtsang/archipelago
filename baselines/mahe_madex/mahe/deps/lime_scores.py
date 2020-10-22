from utils.lime import lime_base
import numpy as np
from utils.general_utils import *
from sklearn.metrics import mean_squared_error


def get_lime_mse(
    Xd,
    Yd,
    max_features=10000,
    kernel_width=0.25,
    weight_samples=True,
    sort=True,
    **kwargs
):
    def kernel(d):
        return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

    distances = get_sample_distances(Xd)["train"]
    if not weight_samples:
        distances = np.ones_like(distances)

    lb = lime_base.LimeBase(kernel_fn=kernel)
    lb_out = lb.explain_instance_with_data(
        Xd["train"], Yd["train"], distances, 0, max_features
    )
    easy_model = lb_out[-1]

    weights = lb_out[-3]
    all_pred = easy_model.predict(Xd["test"])

    Wd = get_sample_weights(Xd, enable=weight_samples, **kwargs)

    mse = mean_squared_error(Yd["test"], all_pred, sample_weight=Wd["test"])

    mse_train = lb_out[2]
    #     print(mse_train, mse)
    #     assert(False)

    return mse
