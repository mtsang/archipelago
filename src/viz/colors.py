import numpy as np
import matplotlib as mpl


def pos_neg_colors(get_rgb=False):
    """
    Based on the Integrated Hessians Code Repo
    """
    color_map_size = 256
    vals = np.ones((color_map_size, 4))
    
    # colors chosen for color-blind
    rgb_a = (210, 110, 105)
    rgb_b = (103, 169, 207)

    vals[: int(color_map_size / 2), 0] = np.linspace(
        rgb_a[0] / 256, 1.0, int(color_map_size / 2)
    )
    vals[: int(color_map_size / 2), 1] = np.linspace(
        rgb_a[1] / 256, 1.0, int(color_map_size / 2)
    )
    vals[: int(color_map_size / 2), 2] = np.linspace(
        rgb_a[2] / 256, 1.0, int(color_map_size / 2)
    )

    vals[int(color_map_size / 2) :, 0] = np.linspace(
        1.0, rgb_b[0] / 256, int(color_map_size / 2)
    )
    vals[int(color_map_size / 2) :, 1] = np.linspace(
        1.0, rgb_b[1] / 256, int(color_map_size / 2)
    )
    vals[int(color_map_size / 2) :, 2] = np.linspace(
        1.0, rgb_b[2] / 256, int(color_map_size / 2)
    )
    cmap = mpl.colors.ListedColormap(vals)
    if get_rgb:
        return rgb_a, rgb_b
    else:
        return cmap
