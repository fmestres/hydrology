import warnings
from typing import Protocol
import numpy as np
from matplotlib.axes import Axes

from src.constants import INITIAL_ABSTRACTION_RATIO, SCS_CN_METHOD_NUMBER, TIME_UNIT, DEFAULT_SERIES_LABEL
from src.exceptions import NotAUnitHydrographError, DeltaTimeMismatchError, GraphTypeMismatchError, \
    NotSupportedStyleError
from src.types import Hyetograph, Hydrograph, TimeSeries


class Plottable(Protocol):
    name: str
    time_series: TimeSeries


def get_convolution(hyetograph: Hyetograph, unit_hydrograph: Hydrograph) -> Hydrograph:
    """
    Returns Hydrograph as the convolution of excess rain hyetograph with unit hydrograph
    :param hyetograph: excess rain hyetograph
    :param unit_hydrograph: unit hydrograph (hydrograph associated to a unit single-pulse hyetograph)
    :return: hydrograph resulting from the convolution, which 'associated_hyetograph' is 'hyetograph'
    :rtype: Hydrograph
    :raises NotAUnitHydrographError if unit_hydrograph is not a unit hydrograph
    :raises DeltaTimeMismatchError if hyetograph and unit_hydrograph have different 'delta_time'
    """

    if not np.array_equal(unit_hydrograph.associated_hyetograph.time_series.data, np.array([1])):
        raise NotAUnitHydrographError("Only a unit hydrograph is supported")

    if hyetograph.time_series.delta_time != unit_hydrograph.time_series.delta_time:
        raise DeltaTimeMismatchError("Unit Hydrograph and Hyetograph pulses must have the same duration")

    hy_series = hyetograph.time_series.data
    uh_series = unit_hydrograph.time_series.data
    delta_time = hyetograph.time_series.delta_time

    hyetograph_length = len(hy_series)
    uh_length = len(uh_series)
    n = uh_length + hyetograph_length - 1
    rain_matrix = np.zeros((n, uh_length))

    for i, pulse in enumerate(hy_series):
        """
        Creates a rainfall matrix ('rain_matrix'): 
        P = [[p1, 0, 0, ...], [p2, p1, 0, ...], [p3, p2, p1, ...], ..., [0, pn, pn-1, ...], ...] 
        where p1, p2, ..., pn are the pulses of the hyetograph
        """
        rain_matrix += np.vstack(
            [np.zeros((i, uh_length)),
             np.eye(uh_length) * pulse,
             np.zeros((hyetograph_length - i - 1, uh_length))]
        )

    series_data = np.dot(rain_matrix, uh_series)
    result_series = TimeSeries(delta_time, series_data, label="m3/s")
    return Hydrograph(f"Convolution of {hyetograph.name} and {unit_hydrograph.name}",
                      result_series,
                      hyetograph)


def get_excess_rainfall_hyetograph_cn(hyetograph: Hyetograph, curve_number: float) -> Hyetograph:
    """
    Returns excess rainfall hyetograph for a given curve number
    (U.S. Soil Conservation Service's curve number method)

    :param hyetograph: gross rainfall hyetograph
    :param curve_number: curve number depending on basin's features
    :returns: excess rainfall hyetograph
    :rtype: Hyetograph
    :raises ValueError if curve_number is not in the range (0, 100]
    """

    if not 0 < curve_number <= 100:
        raise ValueError("Curve number must be between 0 and 100")

    series = hyetograph.time_series.data
    label = hyetograph.time_series.label
    delta_time = hyetograph.time_series.delta_time
    total_rainfall = hyetograph.total_rainfall

    maximum_storage = (100 / curve_number - 1) * SCS_CN_METHOD_NUMBER
    max_i_a = INITIAL_ABSTRACTION_RATIO * maximum_storage
    cum_rainfall = np.cumsum(series)
    initial_abstraction = np.array([max_i_a if p > max_i_a else p for p in cum_rainfall])

    continuous_abstraction = maximum_storage * (cum_rainfall - initial_abstraction) / \
                             (total_rainfall - initial_abstraction + maximum_storage)
    cum_excess_rainfall = cum_rainfall - continuous_abstraction - initial_abstraction
    excess_rainfall = np.trim_zeros(np.diff(cum_excess_rainfall, prepend=0))
    result_series = TimeSeries(delta_time, excess_rainfall, label=label)
    return Hyetograph(f"Excess rainfall hyetograph from {hyetograph.name} for curve number {curve_number}",
                      result_series)


def get_excess_rainfall_hyetograph_phi(hyetograph: Hyetograph, phi: float) -> Hyetograph:
    """
    Returns excess rainfall hyetograph for a given phi (Constant loss rate method)

    :param hyetograph: gross rainfall hyetograph
    :param phi: loss rate (assumed constant)
    :returns: excess rainfall hyetograph
    :rtype: Hyetograph
    :raises ValueError if phi is non-positive

    """
    series = hyetograph.time_series.data
    label = hyetograph.time_series.label
    delta_time = hyetograph.time_series.delta_time

    if phi < 0:
        raise ValueError("Phi must be positive")
    if phi > max(series):
        warnings.warn("Hyetograph has no excess rainfall", UserWarning)
    excess_rainfall = series - phi
    excess_rainfall[excess_rainfall < 0] = 0
    result_series = TimeSeries(delta_time, np.trim_zeros(excess_rainfall), label=label)
    return Hyetograph(f"Excess rainfall hyetograph from {hyetograph.name} for phi = {phi}",
                      result_series)


def plot(graph: Plottable, ax: Axes, style: str = 'bar') -> Axes:
    """
    Plots a graph on the given axes.

    :param graph: graph to plot
    :param ax: Axes object to plot on
    :param style: style of the plot
    :return: Axes object with graph plotted on it
    :rtype: Axes
    """
    series = graph.time_series.data
    total_duration = graph.time_series.total_duration
    delta_t = graph.time_series.delta_time
    label = graph.time_series.label or DEFAULT_SERIES_LABEL
    
    if style == 'bar':
        ax.bar(np.arange(0, total_duration, delta_t), series, width=delta_t, align="edge", label=graph.name)
    elif style == 'line':
        ax.plot(np.arange(0, total_duration, delta_t), series, label=graph.name)
    else:
        raise NotSupportedStyleError(f"Style {style} is not supported. Choose either of 'bar' or 'line'")

    start, stop = ax.get_xlim()
    ax.set(xlabel=F"T [{TIME_UNIT}]",
           ylabel=label,
           xticks=np.arange(0, total_duration, step=delta_t))
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0)
    return ax


def plot_multiple(ax: Axes, *graphs: Plottable) -> Axes:
    """
    Plots graphs of the same type on the same axes.

    :param ax: Axes object to plot on
    :param graphs: Plottable objects to plot, including types 'Hydrograph' and 'Hyetograph'
    :return: Axes object with graphs plotted on it
    :rtype: Axes
    :raises GraphTypeMismatchError if graphs are not of the same type
    """
    if len(set([type(g) for g in graphs])) > 1:
        raise GraphTypeMismatchError("All graphs must be of the same type")
    for graph in graphs:
        plot(graph, ax, style='line')
    ax.legend()
    ax.autoscale()
    return ax
