import warnings
from typing import Protocol
import numpy as np
from matplotlib.axes import Axes

from src.constants import INITIAL_ABSTRACTION_RATIO, SCS_CN_METHOD_NUMBER
from src.exceptions import NotAUnitHydrographError, DeltaTimeMismatchError, GraphTypeMismatchError
from src.types import Hyetograph, Hydrograph


class Plottable(Protocol):
    def plot(self, ax: Axes, style: str) -> Axes:
        ...


def convolution(hyetograph: Hyetograph, unit_hydrograph: Hydrograph) -> Hydrograph:
    """Discharge vs time. uh stands for Unit Hydrograph"""
    if not np.array_equal(unit_hydrograph.hyetograph.series, np.array([1])):
        raise NotAUnitHydrographError("Only a unit hydrograph is supported")

    if hyetograph.delta_t != unit_hydrograph.delta_t:
        raise DeltaTimeMismatchError("Unit Hydrograph and Hyetograph pulses must have the same duration")

    hyetograph_length = len(hyetograph)
    uh_length = len(unit_hydrograph)
    n = uh_length + hyetograph_length - 1
    rain_matrix = np.zeros((n, uh_length))

    for i, pulse in enumerate(hyetograph.series):
        rain_matrix += np.vstack(
            [np.zeros((i, uh_length)),
             np.eye(uh_length) * pulse,
             np.zeros((hyetograph_length - i - 1, uh_length))]
        )

    resulting_series = np.dot(rain_matrix, unit_hydrograph.series)
    return Hydrograph(hyetograph.delta_t, resulting_series, hyetograph)


def _get_excess_rainfall_hyetograph_cn(hyetograph: Hyetograph, curve_number: float) -> Hyetograph:
    """
    Returns excess rainfall hyetograph for a given curve number
    (U.S Soil Conservation Service's curve number method)
    """
    if not 0 < curve_number <= 100:
        raise ValueError("Curve number must be between 0 and 100")
    maximum_storage = (100 / curve_number - 1) * SCS_CN_METHOD_NUMBER
    max_i_a = INITIAL_ABSTRACTION_RATIO * maximum_storage
    cum_rainfall = np.cumsum(hyetograph.series)
    initial_abstraction = np.array([max_i_a if p > max_i_a else p for p in cum_rainfall])
    total_rainfall = hyetograph.total_rainfall
    continuous_abstraction = maximum_storage * (cum_rainfall - initial_abstraction) / \
                             (total_rainfall - initial_abstraction + maximum_storage)
    cum_excess_rainfall = cum_rainfall - continuous_abstraction - initial_abstraction
    excess_rainfall = np.trim_zeros(np.diff(cum_excess_rainfall, prepend=0))
    return Hyetograph(hyetograph.delta_t, excess_rainfall)


def _get_excess_rainfall_hyetograph_phi(hyetograph: Hyetograph, phi: float) -> Hyetograph:
    """
    Returns excess rainfall hyetograph for a given phi (Constant loss rate method)
    """
    if phi < 0:
        raise ValueError("Phi must be positive")
    if phi > max(hyetograph.series):
        warnings.warn("Hyetograph has no excess rainfall", UserWarning)
    excess_rainfall = hyetograph.series - phi
    excess_rainfall[excess_rainfall < 0] = 0
    return Hyetograph(hyetograph.delta_t, np.trim_zeros(excess_rainfall))


def plot_graph_comparison(ax: Axes, *graphs: Plottable) -> Axes:
    """
    Plots graphs of the same type on the same axes. \n
    ax: Axes object to plot on \n
    graphs: Plottable objects to plot, including types 'Hydrograph' and 'Hyetograph'
    """
    if len(set([type(g) for g in graphs])) > 1:
        raise GraphTypeMismatchError("All graphs must be of the same type")
    for graph in graphs:
        graph.plot(ax, style='line')
    ax.legend()
    ax.autoscale()
    return ax
