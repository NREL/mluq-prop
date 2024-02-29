import os
import numpy as np
import sys
sys.path.append('util')
from fileManagement import *
from prettyPlot.plotting import plt, pretty_labels, pretty_legend


dataFolder = 'data'

cluster_list = getClusterList(dataFolder)
data_list = [np.load(os.path.join(dataFolder, f'frechetDistRef_{cluster}.npz')) for cluster in cluster_list]

ndim = data_list[0]['loc'].shape[1]

figureFolder = 'Figures'

def color_clust(i,cluster_list):
    color = str(i/(len(cluster_list)))
    return color


quantities = ['mean', 'epist', 'aleat']

for quantity in quantities:
    os.makedirs(figureFolder, exist_ok=True)
    os.makedirs(os.path.join(figureFolder, quantity), exist_ok=True)
    for idim in range(ndim):
        fig = plt.figure()
        for i, (clust, data) in enumerate(zip(cluster_list, data_list)):
            color = color_clust(i, cluster_list)
            plt.plot(data['loc'][:,idim], data[quantity][:],'o', color=color, linewidth=3, label=f'cl. {clust}')
        pretty_legend()
        pretty_labels(f'feature {idim}', quantity, 14)
        plt.savefig(os.path.join(figureFolder,quantity,f'dim{idim}.png'))
        plt.close()


