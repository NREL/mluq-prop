import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from prettyPlot.plotting import pretty_cbar, pretty_labels, pretty_legend

def PlotScatterMeanWithUncertainty(simparams, true, predicted, uncertainties, utype:str):
    """Make a 45-degree scatter plot of true vs. predicted shaded by uncertainty.
    WARNING! This function is fragile and not generalized.

    Args:
        simparams: Input deck.
        true: True data values.
        predicted: Model predictions.
        uncertainties: Uncertainties used to shade the scatter plot.
        utype (str): "Epistemic" or "Aleatoric"
    """
    # Plot predictive Uncertainty
    plt.figure()
    plt.scatter(true, predicted, c=uncertainties, cmap='summer')
    # plt.xlim([0, np.max(Ytest)])
    # plt.ylim([0, np.max(preds_mean)])
    cbar=pretty_cbar(label=f"{utype} Uncertainty $[-]$", fontsize=18) 
    pretty_labels(xlabel=r"True $\widetilde{\chi}_{\rm SFS} / \chi_{\rm lam} [-]$", ylabel=r"Predicted $\widetilde{\chi}_{\rm SFS} / \chi_{\rm lam} [-]$", fontsize=18)
    plt.savefig(os.path.join(simparams.savedir, f"{utype.lower()}_uncertainty_scatter_{simparams.filename}.png"))
    plt.close()


def PlotLogHexScatter_simp(true:np.array, predicted:np.array, xlab:str, ylab:str):
    """Make a 45-degree log-scale scatter plot of true vs precited data.
    WARNING! This function is fragile and not generalized.

    Args:
        true (np.array): True data values.
        predicted (np.array): Model predictions.
        xlab (str): x label.
        ylab (str): y label.
    """
    plt.figure()
    ax = plt.gca()
    ax.axline((0, 0), slope=1, c='k', lw=2.0, ls='--')
    plt.hexbin(true, predicted, bins='log', cmap=plt.matplotlib.cm.binary, mincnt=1)
    # cbar=pretty_cbar(label=r"$\rm{log}_{10}(N)$", fontsize=15)
    cbar=pretty_cbar(label=r"Amount of Data", fontsize=15)
    #plt.colorbar(label="Amount of Data")
    pretty_labels(xlabel=xlab, ylabel=ylab, fontsize=15)

def PlotLogHexScatter(simparams, true:np.array, predicted:np.array):
    """Make a 45-degree log-scale scatter plot of true vs precited data.
    WARNING! This function is fragile and not generalized.

    Args:
        simparams: Input deck.
        true (np.array): True data values.
        predicted (np.array): Model predictions.
    """
    PlotLogHexScatter_simp(true, predicted, xlab=r"True $\widetilde{\chi}_{\rm SFS} / \chi_{\rm lam} [-]$", ylab=r"Predicted $\widetilde{\chi}_{\rm SFS} / \chi_{\rm lam} [-]$")
    plt.savefig(os.path.join(simparams.savedir, f"hexscatter_{simparams.filename}.pdf"))
    plt.close()


def PlotInducingScatter(simparams, true:np.array, predicted:np.array, scale:np.array, plotname:str, cbarname:str, errorbars:bool=False, uipseb=None, xlab='', ylab=''):
    """Make a 45-degree scatter plot of true vs. predicted shaded by a scale.

    Args:
        simparams: Input Deck.
        true (np.array): True data.
        predicted (np.array): Model predictions.
        scale (np.array): Either uncertainty or probability.
        plotname (str): Namestring for saved plot.
        cbarname (str): Colorbar name.
        errorbars (bool, optional): Whether or not to include error bars. Defaults to False.
        uipseb (_type_, optional): Errorbar values. Defaults to None.
    """
    plt.figure()
    ax = plt.gca()
    ax.axline((0, 0), slope=1, c='k', lw=2.0, ls='--')
    plt.scatter(true, predicted, c=scale, cmap='summer')
    cbar=pretty_cbar(label=f"{cbarname}", fontsize=18)
    #cbar = plt.colorbar()
    #cbar.set_label(f"{cbarname}")
    if errorbars:
        if uipseb is None:
            raise ValueError("uipseb must be provided if errorbars is True.")
        eb = uipseb.tolist()
        plt.errorbar(true, predicted, yerr=eb, fmt=None, marker=None, mew=0)
    pretty_labels(xlabel=f"True {xlab}", ylabel=f"Predicted {ylab}", fontsize=18)
    plt.locator_params(axis='x', nbins=4)
    plt.tight_layout()
    plt.savefig(os.path.join(simparams.savedir, f"{plotname}_scatter_{simparams.filename}.pdf"))
    plt.close()



def PlotConfidenceInterval_simp(x:np.array, y:np.array, xbin:np.array, ybindata:np.array, ybinpred:np.array, ybinepi:np.array, ybinale:np.array, name:str):
    plt.figure()
    plt.hexbin(x, y, bins='log', cmap=plt.matplotlib.cm.binary, mincnt=1)
    plt.plot(xbin, ybindata, 'k-', lw=2.0, label="Validation Data")
    plt.plot(xbin, ybinpred, 'b-', lw=2.0, label="Predictive Mean")
    # plt.plot(xbinpred, ybinlrm, 'r-.', lw=2.0, label="Linear Relaxation Model")
    ax = plt.gca()
    ax.fill_between(xbin, ybinpred - (ybinepi + ybinale), ybinpred + (ybinepi + ybinale), color="lightblue", alpha=0.5, label="Predictive Uncertainty")
    # ax.fill_between(xbindata, ybinpred - ybinepistd, ybinpred + ybinepistd, color="lightcoral", alpha=0.5, label="Epistemic Uncertainty")
    cbar = pretty_cbar(label=r"$\rm{log}_{10}(N)$", fontsize=18)
    #plt.colorbar(label="Amount of Data")
    pretty_legend()
    plt.locator_params(axis='x', nbins=4)
    pretty_labels(xlabel=name, ylabel=r"$\widetilde{\chi}_{\rm SFS} / \chi_{\rm lam} [-]$", fontsize=18)

def PlotConfidenceInterval(x:np.array, y:np.array, xbin:np.array, ybindata:np.array, ybinpred:np.array, ybinepi:np.array, ybinale:np.array, name:str, shortname:str, simparams):
    PlotConfidenceInterval_simp(x,y,xbin,ybindata, ybinpred, ybinepi, ybinale, name)
    plt.savefig(os.path.join(simparams.savedir, f"condmean_confidence_{shortname}.pdf"))
    plt.close()


def PlotConditionalDataLogHex(x:np.array, y:np.array, xbin:np.array, ybindata:np.array, name:str, shortname:str, simparams):
    plt.figure()
    plt.hexbin(x, y, bins='log', cmap=plt.matplotlib.cm.binary, mincnt=1)
    plt.plot(xbin, ybindata, 'b--', lw=2.0, label="Validation Data Mean")
    cbar = pretty_cbar(label=r"$\rm{log}_{10}(N)$", fontsize=18)
    pretty_legend()
    plt.locator_params(axis='x', nbins=4)
    pretty_labels(xlabel=name, ylabel=r"$\widetilde{\chi}_{\rm SFS} / \chi_{\rm lam} [-]$", fontsize=18)
    plt.savefig(os.path.join(simparams.savedir, f"conditional_data_hex_{shortname}.pdf"))
    plt.close()


def PlotCredibleInterval(xbin, ybindata, xbinpred, ybinpred, ptilelo, ptilehi, name, shortname, simparams):
    plt.figure()
    plt.plot(xbin, ybindata, 'k-', lw=2.0, label="Validation Data")
    plt.plot(xbin, ybinpred, 'b-', lw=2.0, label="Predictive Mean")
    # plt.plot(xbinpred, ybinlrm, 'r-.', lw=2.0, label="Linear Relaxation Model")
    ax = plt.gca()
    ax.fill_between(xbinpred, ptilelo, ptilehi, color="lightblue", alpha=0.5, label="Predictive Uncertainty")
    # ax.fill_between(xbinpred, ybinepipredptilelo, ybinepipredptilehi, color="lightcoral", alpha=0.5, label="Epistemic Uncertainty")
    pretty_legend()
    pretty_labels(xlabel=name, ylabel=r"$\widetilde{\chi}_{\rm SFS} / \chi_{\rm lam} [-]$", fontsize=16)
    plt.savefig(os.path.join(simparams.savedir, f"condmean_credible_{shortname}.pdf"))
    plt.close()


def PlotCredibleIntervalInset(xbin, ybindata, xbinpred, ybinpred, ptilelo, ptilehi, epiptilelo, epiptilehi, name, shortname, simparams):
    plt.figure()
    plt.plot(xbin, ybindata, 'k-', lw=2.0, label="Validation Data")
    plt.plot(xbin, ybinpred, 'b-', lw=2.0, label="Predictive Mean")
    # plt.plot(xbinpred, ybinlrm, 'r-.', lw=2.0, label="Linear Relaxation Model")
    ax = plt.gca()
    ax.fill_between(xbinpred, ptilelo, ptilehi, color="lightblue", alpha=0.5, label="Predictive Uncertainty")
    ax.fill_between(xbinpred, epiptilelo, epiptilehi, color="lightcoral", alpha=0.5, label="Epistemic Uncertainty")
    
    # attempt to make inset
    x1, x2, y1, y2 = 0.4, 0.6, 0.075, 0.175
    
    axins = ax.inset_axes(
    [0.5, 0.5, 0.47, 0.47],
    xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    axins.plot(xbin, ybindata, 'k-', lw=2.0)
    axins.plot(xbin, ybinpred, 'b-', lw=2.0)
    axins.fill_between(xbinpred, ptilelo, ptilehi, color="lightblue", alpha=0.5)
    axins.fill_between(xbinpred, epiptilelo, epiptilehi, color="lightcoral", alpha=0.5)
    ax.indicate_inset_zoom(axins, edgecolor="black")
    
    plt.savefig(os.path.join(simparams.savedir, f"inset_condmean_credible_{shortname}.pdf"))
    plt.close()
    # pretty_legend()
    # pretty_labels(xlabel=name, ylabel=r"$\widetilde{\chi}_{\rm SFS} / \chi_{\rm lam} [-]$", fontsize=16)
    # plt.savefig(os.path.join(simparams.savedir, f"inset_condmean_credible_{shortname}.pdf"))
    # plt.close()


def PlotConditionalStdDev(xbin, ybinstd, name, shortname, simparams, savename, colors=None, labels=None):
    plt.figure()
    if not isinstance(xbin, list):
        plt.plot(xbin, ybinstd, 'k-', lw=3.0, label="Confidence Interval")
    else:
        for (xb, yb, n, sn, lab, col) in zip(xbin, ybinstd, name, shortname, labels, colors):
            plt.plot(xb, yb, f'{col}-', lw=3.0, label=lab)
        pretty_legend(fontsize=16)
    pretty_labels(xlabel=name, ylabel=r"$\chi_{\rm SFS} / \chi_{\rm lam} [-]$", fontsize=16)
    plt.savefig(os.path.join(simparams.savedir, f"std_{savename}_confidence_{shortname}.pdf"))
    plt.close()
