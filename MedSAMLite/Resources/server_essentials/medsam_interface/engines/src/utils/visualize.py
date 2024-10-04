import numpy as np
from matplotlib import pyplot as plt


def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, edgecolor="blue"):
    """
    show bounding box on the image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    ax : matplotlib.axes.Axes
        axes to plot the bounding box
    edgecolor : str
        color of the bounding box
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0, 0, 0, 0), lw=2)
    )


def visualize_output(img, boxes, segs, save_file=None):
    _, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    ax[1].imshow(img)
    ax[0].set_title("Input")
    ax[1].set_title("Segmentation")
    ax[0].axis("off")
    ax[1].axis("off")

    for i, box in enumerate(boxes):
        color = np.random.rand(3)
        box_viz = box
        mask = (segs == i + 1).astype(np.uint16)
        show_box(box_viz, ax[0], edgecolor=color)

        if np.max(mask) > 0:
            show_box(box_viz, ax[1], edgecolor=color)
            show_mask(mask, ax[1], mask_color=color)

    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, dpi=300)
        plt.close()
    else:
        plt.show()
