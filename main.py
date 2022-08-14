import numpy as np
import matplotlib.pyplot as plt

from src.functions import convolution, _get_excess_rainfall_hyetograph_cn, _get_excess_rainfall_hyetograph_phi, \
    plot_graph_comparison
from src.plot_styles import PlotStyles
from src.types import Hyetograph, Hydrograph


def main():
    plt.style.use(PlotStyles.seaborn_darkgrid.value)
    fig, (hye, hyd, comparison) = plt.subplots(3, 1,
                                   sharex='all',
                                   dpi=200,
                                   gridspec_kw={'height_ratios': [1, 1, 1]})

    fig.suptitle(f"Hidrograma de escurrimiento directo (Δt = 4 hs)", fontsize=14)

    uh = Hydrograph(4, np.array([
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
    ]))  # unit hydrograph
    hy = Hyetograph(4, np.array([
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
        5.4,
g
    ]))  # net hyetograph
    hy_net = _get_excess_rainfall_hyetograph_phi(hy, 124)
    hy_net.plot(hyd)
    hed = convolution(hy_net, uh)
    hed.hyetograph.plot(hye)
    plt.tick_params(axis='x', which='both', labelsize=8, labelrotation=90)
    plot_graph_comparison(comparison, hy, hy_net, uh, hed)
    plt.show()


if __name__ == "__main__":
    main()

