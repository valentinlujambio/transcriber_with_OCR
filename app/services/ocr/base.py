from abc import ABC, abstractmethod
import numpy as np


class OCRStrategy(ABC):
    @abstractmethod
    def extract_text(self, image_region: np.ndarray) -> str:
        ...
