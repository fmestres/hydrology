import numpy as np
import matplotlib.pyplot as plt

from src.functions import get_convolution, get_excess_rainfall_hyetograph_cn, plot

from src.types import Hyetograph, Hydrograph, TimeSeries


def main():
    """Example"""
    plot_conf = {
        "sharex": 'all',
        "dpi": 300,
        "gridspec_kw": {'height_ratios': [1, 2]}
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, **plot_conf)

    fig.suptitle(f"Hidrograma de escurrimiento directo (Δt = 4 hs)", fontsize=14)

    uh_ts = TimeSeries(4, np.array([
        0.1454721557,
        0.2909443113,
        0.436416467,
        0.5818886226,
        0.7273607783,
        0.872832934,
        1.01830509,
        1.163777245,
        1.194755784,
        1.080456234,
        0.9661566827,
        0.8518571318,
        0.7375575809,
        0.62325803,
        0.5089584792,
        0.3946589283,
        0.2803593774,
        0.1660598265,
        0.05176027566
    ]), 'Q [m3/s]')  # time series for unit hydrograph

    hy_ts = TimeSeries(4, np.array([
        0.3,
        2.4,
        5.7,
        12.5,
        126.1,
        28.2,
        13.8,
        9.6,
        7.5,
        6.3,
        5.4
    ]), 'P [mm]')  # time series for net hyetograph

    unit_hydrograph = Hydrograph("Unit Hydrograph", uh_ts)
    gross_rainfall_hyetograph = Hyetograph("Hyetograph", hy_ts)

    excess_rainfall_hyetograph = get_excess_rainfall_hyetograph_cn(
                                    gross_rainfall_hyetograph,
                                    curve_number=70
                                )

    hed = get_convolution(excess_rainfall_hyetograph, unit_hydrograph)

    plot(hed.associated_hyetograph, ax1)  # Same as 'excess_rainfall_hyetograph'
    plot(hed, ax2, 'line')

    plt.tick_params(axis='x', which='both', labelsize=8, labelrotation=90)

    plt.show()


if __name__ == "__main__":
    main()

