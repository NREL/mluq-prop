from prettyPlot.plotting import plt, pretty_labels, pretty_legend, matplotlib
import numpy as np

def plotResult(Result, Sim):

    if Sim["Plot"]:
        fig = plt.figure()
        if Sim["Simulation name"] == "KS":
            plt.imshow(
                Result["uu"][:, :, 0],
                origin="lower",
                cmap="jet",
                interpolation="nearest",
                aspect="auto",
                extent=[0, Sim["Lx/pi"] * np.pi, 0, Sim["Tf"]],
            )
        plt.locator_params(axis="x", nbins=5)
        plt.locator_params(axis="y", nbins=5)
        # plt.colorbar()
        cbar = plt.colorbar()
        if "clabel" in Sim:
            cbar.set_label(Sim["clabel"])
        else:
            cbar.set_label(r"$\xi_i$")
        ax = cbar.ax
        text = ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(
            family="times new roman", weight="bold", size=25
        )
        text.set_font_properties(font)
        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_weight("bold")
            l.set_family("serif")
            l.set_fontsize(18)
        fig.tight_layout()
        if "plotTitle" in Sim:
            pretty_labels("x", "t", 18, title=Sim["plotTitle"])
        else:
            pretty_labels("x", "t", 18)

            

        if Sim["Simulation name"] in ["KS"]:
            fig = plt.figure()
            plt.plot(
                Result["uu"][-1, :, 0], linewidth=3, color="k", label="end"
            )
            pretty_labels("x", "u", 14)
            pretty_legend()

        plt.show()


def postProc_kse(Result, Sim):
    plotResult(Result, Sim)
