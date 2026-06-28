from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

Point = tuple[float, float]
Line = tuple[Point, Point]
Rect = tuple[float, float, float, float]  # x, y, w, h


class DebugPlotter:
    """Utilidades de visualización para debug del pipeline."""

    def __init__(
        self,
        save: bool = True,
        show: bool = False,
        output_dir: str | Path = "debug",
        figsize: tuple[float, float] = (8, 10),
        dpi: int = 120,
    ) -> None:
        self.save = save
        self.show = show
        self.output_dir = Path(output_dir)
        self.figsize = figsize
        self.dpi = dpi

    def _new_canvas(self, image: np.ndarray, title: str):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
        return fig, ax

    def _finalize(self, fig, title: str) -> None:
        if self.save:
            path = self.output_dir / f"{title}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches="tight", dpi=self.dpi)
        if self.show:
            plt.show()
        plt.close(fig)

    def plot_image(self, image: np.ndarray, title: str) -> None:
        fig, _ = self._new_canvas(image, title)
        self._finalize(fig, title)

    def plot_points(
        self,
        image: np.ndarray,
        points: Sequence[Point],
        title: str,
        color: str = "red",
        size: float = 20,
        marker: str = "o",
    ) -> None:
        fig, ax = self._new_canvas(image, title)
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        ax.scatter(pts[:, 0], pts[:, 1], c=color, s=size, marker=marker)
        self._finalize(fig, title)

    def plot_lines(
        self,
        image: np.ndarray,
        lines: Iterable[Line],
        title: str,
        color: str = "red",
        width: float = 1.5,
    ) -> None:
        fig, ax = self._new_canvas(image, title)
        for (x1, y1), (x2, y2) in lines:
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=width)
        self._finalize(fig, title)

    def plot_marks(
        self,
        image: np.ndarray,
        groups: Iterable[tuple[Iterable[Rect], str]],
        title: str,
        width: float = 2.0,
    ) -> None:
        """Dibuja varios grupos de rectangulos (cada uno con su color) sobre la imagen."""
        fig, ax = self._new_canvas(image, title)
        for rectangles, color in groups:
            for x, y, w, h in rectangles:
                ax.add_patch(
                    Rectangle(
                        (x, y),
                        w,
                        h,
                        edgecolor=color,
                        facecolor="none",
                        linewidth=width,
                    )
                )
        self._finalize(fig, title)

    def plot_rectangles(
        self,
        image: np.ndarray,
        rectangles: Iterable[Rect],
        title: str,
        color: str = "red",
        width: float = 1.5,
        fill: bool = False,
    ) -> None:
        fig, ax = self._new_canvas(image, title)
        for x, y, w, h in rectangles:
            ax.add_patch(
                Rectangle(
                    (x, y),
                    w,
                    h,
                    edgecolor=color,
                    facecolor=color if fill else "none",
                    linewidth=width,
                    alpha=0.3 if fill else 1.0,
                )
            )
        self._finalize(fig, title)


debug_plotter = DebugPlotter()
