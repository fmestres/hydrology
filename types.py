from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.exceptions import DeltaTimeMismatchError, SeriesDimensionError


@dataclass
class TimeSeries:
    """
    Representation of a time series graph \n
    'delta_time': time between two consecutive observations in 'data' \n
    'data': 1-D numpy array containing the observations of the time series
    """

    delta_time: int
    data: NDArray
    label: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.data.ndim != 1:
            raise SeriesDimensionError("Time series data must be 1-D")

    @cached_property
    def total_duration(self) -> int:
        return len(self.data) * self.delta_time


@dataclass
class Hyetograph:
    """
    Representation of a hyetograph (rainfall distribution over time) \n
    'name': name of the hyetograph \n
    'time_series': time series from which the hyetograph is created
    """
    name: str
    time_series: TimeSeries

    @cached_property
    def total_rainfall(self) -> float:
        return self.time_series.data.sum()


@dataclass
class Hydrograph:
    """
    Representation of a hydrograph (discharge distribution over time) \n
    'name': name of the hydrograph \n
    'time_series': time series from which the hydrograph is created \n
    'associated_hyetograph': hyetograph of which the hydrograph is the response
    """
    name: str
    time_series: TimeSeries
    associated_hyetograph: Optional[Hyetograph] = field(default=None)

    def __post_init__(self):
        if self.associated_hyetograph is None:
            """if no hyetograph is associated, create a unit hyetograph"""
            self.associated_hyetograph = Hyetograph(
                f"Unit hyetograph of {self.name}",
                TimeSeries(self.time_series.delta_time, np.array([1]))
            )
        elif self.associated_hyetograph.time_series.delta_time != self.time_series.delta_time:
            raise DeltaTimeMismatchError(f"Delta time of hyetograph {self.associated_hyetograph.name} and "
                                         f"hydrograph {self.name} do not match")

    @property
    def total_volume(self):
        return self.time_series.data.sum() * self.time_series.delta_time




