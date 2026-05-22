import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def plot_image(image: np.ndarray, title: str) -> None:
    """
    Plota una imagen y la guarda en el directorio debug para debugging.
    """
    
    debug_path = Path("debug") / f"{title}.png"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(debug_path, bbox_inches="tight", dpi=120)
    plt.close(fig)