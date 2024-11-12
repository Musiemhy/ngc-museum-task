import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

colors = ["red"]
# post-process learning curve data
y = (np.load("exp/efe.npy")) ## EFE measurements
x_iter = np.asarray(list(range(0, y.shape[0])))
## make the plots with matplotlib
fontSize = 20
plt.plot(x_iter, y, '-', color=colors[0])
plt.xlabel("Epoch", fontsize=fontSize)
plt.ylabel("Free Energy", fontsize=fontSize)
plt.grid()
## construct the legend/key
freeEnergy = mpatches.Patch(color=colors[0], label='Dev EFE')
plt.legend(handles=[freeEnergy], fontsize=13, ncol=2,borderaxespad=0, frameon=False,
           loc='upper center', bbox_to_anchor=(0.5, -0.175))
plt.tight_layout()
plt.savefig("exp/pcn_free_energy_curves.jpg") ## save plot to disk
plt.clf()