import os
import numpy as np
import sys
sys.path.append('util')
from prettyPlot.plotting import pretty_labels, pretty_legend, plt
from matplotlib.ticker import MaxNLocator

conv_result = np.load('convergence.npz')

cluster_list = list(conv_result['cluster'])
mean_pred = list(conv_result['mean'])
aleat_pred = list(conv_result['aleat'])
epist_pred = list(conv_result['epist'])

index_max = cluster_list.index(max(cluster_list))
assert index_max == len(cluster_list)-1
ref_mean = mean_pred[-1]
ref_aleat = aleat_pred[-1]
ref_epist = epist_pred[-1]

error_mean=[]
error_aleat=[]
error_epist=[]


for i, clust in enumerate(cluster_list[:-1]):
    error_mean.append(np.linalg.norm(mean_pred[i]-ref_mean))
    error_aleat.append(np.linalg.norm(aleat_pred[i]-ref_aleat))
    error_epist.append(np.linalg.norm(epist_pred[i]-ref_epist))


figureFolder = 'Figures'

fig = plt.figure()
plt.plot(cluster_list[:-1], error_mean, 'o', color='k', label='mean.')
plt.plot(cluster_list[:-1], error_aleat, 'o', color='r', label='aleat.')
#plt.plot(cluster_list[:-1], error_epist, 'o', color='b', label='epist')
pretty_legend()
ax = plt.gca()
plt.xticks(cluster_list[:-1])
ax.set_yscale('log')
pretty_labels('# cluster', 'L2 err', 14) 
plt.savefig(os.path.join(figureFolder, 'convergence.png'))
plt.close()

fig = plt.figure()
plt.plot(cluster_list[:-1], error_mean/np.mean(mean_pred), 'o', color='k', label='mean.')
plt.plot(cluster_list[:-1], error_aleat/np.mean(aleat_pred), 'o', color='r', label='aleat.')
#plt.plot(cluster_list[:-1], error_epist/np.mean(epist_pred), 'o', color='b', label='epist')
pretty_legend()
ax = plt.gca()
plt.xticks(cluster_list[:-1])
ax.set_yscale('log')
pretty_labels('# cluster', 'L2 err', 14) 
plt.savefig(os.path.join(figureFolder, 'convergence_ref.png'))
plt.close()
