"""
benchmark.variants.base_variant
===============================
Abstract base class defining the acoustic localization algorithm interface.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np


class AcousticAlgorithmVariant(ABC):
    """
    Standard interface for an acoustic localization algorithm variant.
    """

    def __init__(self, name: str, description: str, parameters: Dict[str, Any] = None) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters or {}

    @abstractmethod
    def localize(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute acoustic localization on one 16-channel audio frame.
        Input:
            frame: (samples, 16) array of float samples
        Output:
            heatmap: 2D array (grid_el, grid_az) normalized to [0, 1]
            peak_dict: Dictionary containing:
                "azimuth": float (deg)
                "elevation": float (deg)
                "value": float (peak amplitude)
                "db": float (sound pressure level)
                "valid": bool (noise floor validity)
        """
        pass

    @abstractmethod
    def reset_state(self) -> None:
        """Reset internal temporal state (e.g. smoothed CSM history)."""
        pass
