from sklearn.datasets import make_moons
from sbo import soft_brownian_offset
from prettyPlot.plotting import plt, pretty_labels

X, _ = make_moons(n_samples=600, noise=.08)
#X_ood = soft_brownian_offset(X, d_min=.35, d_off=.24, n_samples=1200, softness=0)
#X_ood = soft_brownian_offset(X, d_min=.35, d_off=.024, n_samples=1200, softness=0) #low support for OOD points
#X_ood = soft_brownian_offset(X, d_min=.035, d_off=.24, n_samples=1200, softness=0) #dmin is distance to the data distribution
X_ood = soft_brownian_offset(X, d_min=.1, d_off=.24, n_samples=1200, softness=0) #Softness generated points in holes of the hull

markersize = 10
fig = plt.figure()
plt.plot(X[:,0],X[:,1], 'o', color='k', markersize=markersize)
plt.plot(X_ood[:,0],X_ood[:,1], 'x', color='r', markersize=markersize)
pretty_labels("","",14)
plt.show()

