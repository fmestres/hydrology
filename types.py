from dataclasses import dataclass, Field
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from src.exceptions import NotSupportedStyleError, DeltaTimeMismatchError, SeriesDimensionError


class Hyetograph:
    def __init__(self, delta_t: int, series: NDArray, name: str = 'Hietograma'):
        """Hyetograph. delta_t is the duration of each pulse."""
        self.delta_t = delta_t
        self.series = series
        self.total_duration = len(series) * delta_t
        self.name = name

    def __str__(self):
        return f"delta_t: {self.delta_t}, series: {self.series}"

    def __len__(self):
        return len(self.series)

    @property
    def total_rainfall(self) -> float:
        return np.sum(self.series)

    def plot(self, ax: Axes, style: str = 'bar') -> Axes:
        """Plot hyetograph on ax"""
        total_duration = self.total_duration
        delta_t = self.delta_t
        if style == 'bar':
            ax.bar(np.arange(0, total_duration, delta_t), self.series, width=delta_t, align="edge", label=self.name)
        elif style == 'line':
            ax.plot(np.arange(total_duration), self.series, label=self.name)
        else:
            raise NotSupportedStyleError(f"Style {style} is not supported")

        ax.set(xlabel="T [s]",
               ylabel="P [mm]",
               xticks=np.arange(delta_t, total_duration + delta_t, delta_t))
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)
        return ax




class Hydrograph:
    def __init__(self, delta_t: int, series: NDArray, hyetograph: Hyetograph = None, name: str = 'Hidrograma'):
        """
        Hydrograph. Default is unit hydrograph (hyetograph.series = [1]).  \n
        delta_t must be the same as hyetograph.delta_t
        """
        if hyetograph and delta_t != hyetograph.delta_t:
            raise DeltaTimeMismatchError("delta_t must be the same as hyetograph.delta_t")
        self.hyetograph = hyetograph if hyetograph else Hyetograph(delta_t, np.array([1]))
        self.delta_t = delta_t
        self.series = series
        self.total_duration = len(series) * delta_t
        self.name = name

    def __str__(self):
        return f"Hydrograph (delta_t: {self.delta_t}, series: {self.series})"

    def __len__(self):
        return len(self.series)

    def plot(self, ax: Axes, style: str = 'bar') -> Axes:
        """Plots hydrograph on ax. (style: 'bar' or 'line')"""
        total_duration = self.total_duration
        delta_t = self.delta_t
        if style == 'bar':
            ax.bar(np.arange(0, total_duration, delta_t), self.series, width=delta_t, align="edge", label=self.name)
        elif style == 'line':
            ax.plot(np.arange(0, total_duration, delta_t), self.series, label=self.name)
        else:
            raise NotSupportedStyleError(f"Plot style {style} is not supported")
        ax.set(xlabel="T [s]",
               ylabel="Q [mm]",
               xticks=np.arange(delta_t, total_duration + delta_t, delta_t))
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)
        return ax


