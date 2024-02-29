import os
import numpy as np
import sys
from prettyPlot.plotting import pretty_labels, pretty_legend, plt
from matplotlib.ticker import MaxNLocator

dist_result = np.load('data/inducingDist.npz')

figureFolder = 'Figures'
os.makedirs(figureFolder, exist_ok=True)

fig = plt.figure()
plt.errorbar(dist_result['nCl'], dist_result['mean_uips'], yerr=dist_result['std_uips'], linewidth=3, ecolor='b', elinewidth=3, barsabove=True, label='uips')
plt.errorbar(dist_result['nCl'], dist_result['mean_all'], yerr=dist_result['std_all'], linewidth=3, ecolor='k', elinewidth=3, barsabove=True, label='all')
pretty_legend()
pretty_labels('# cluster', 'Distance criterion', 14) 
plt.savefig(os.path.join(figureFolder, 'distErr.png'))
plt.close()


fig = plt.figure()
plt.plot(dist_result['nCl'], dist_result['mean_uips'], linewidth=3, color='b', label='uips')
plt.plot(dist_result['nCl'], dist_result['mean_all'], linewidth=3, color='k', label='all')
pretty_legend()
pretty_labels('# cluster', 'Distance criterion', 14) 
plt.savefig(os.path.join(figureFolder, 'dist.png'))
plt.close()
