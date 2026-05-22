import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12})


def plot_comparison_images(images, titles, save_path=None, show=True):
    n = len(images)
    vmin, vmax = np.percentile(images[0], [1, 99])
    plt.figure(figsize=(4 * n, 4))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, n, i + 1)
        plt.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=450, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison_grid(target, preds_by_method, psnr_by_method, col_labels,
                        sparse_views, save_path=None, show=True):
    """Grid layout: one row per V, columns = [Target, method_1, method_2, ...].

    Row 0 (first V): title shows method name + PSNR.
    Rows 1+: title shows only PSNR.

    Args:
        target: ndarray, ground truth region.
        preds_by_method: dict method_name -> {V: ndarray}.
        psnr_by_method: dict method_name -> {V: psnr_value}.
        col_labels: list of method names for column ordering after Target.
        sparse_views: list of V values (row labels).
        save_path: optional path to save figure.
        show: whether to call plt.show().
    """
    n_rows = len(sparse_views)
    n_cols = 1 + len(col_labels)  # target + methods
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    vmin, vmax = np.percentile(target, [1, 99])

    for r, V in enumerate(sparse_views):
        is_first_row = (r == 0)

        # Column 0: target
        ax = axes[r, 0]
        ax.imshow(target, cmap="gray", vmin=vmin, vmax=vmax)
        if is_first_row:
            ax.set_title("Target", fontsize=14, fontweight="bold")
        ax.axis("off")
        ax.set_ylabel(f"V={V}", fontsize=15, rotation=90, labelpad=15, va="center")

        for c, method in enumerate(col_labels):
            ax = axes[r, c + 1]
            img = preds_by_method[method][V]
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)

            psnr_val = psnr_by_method[method].get(V, None)
            if is_first_row:
                title = method
                if psnr_val is not None:
                    title += f"\nPSNR={psnr_val:.2f}"
                ax.set_title(title, fontsize=14, fontweight="bold")
            else:
                if psnr_val is not None:
                    ax.set_title(f"PSNR={psnr_val:.2f}", fontsize=12)
            ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=450, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
