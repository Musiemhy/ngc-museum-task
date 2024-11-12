import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

colors = ["red", "blue"]
# post-process learning curve data
y = (1.0 - np.load("exp/trAcc.npy")) * 100. ## training measurements
vy = (1.0 - np.load("exp/acc.npy")) * 100. ## dev measurements
x_iter = np.asarray(list(range(0, y.shape[0])))
## make the plots with matplotlib
fontSize = 20
plt.plot(x_iter, y, '-', color=colors[0])
plt.plot(x_iter, vy, '-', color=colors[1])
plt.xlabel("Epoch", fontsize=fontSize)
plt.ylabel("Classification Error", fontsize=fontSize)
plt.grid()
## construct the legend/key
loss = mpatches.Patch(color=colors[0], label='Train Acc')
vloss = mpatches.Patch(color=colors[1], label='Dev Acc')
plt.legend(handles=[loss, vloss], fontsize=13, ncol=2,borderaxespad=0, frameon=False,
           loc='upper center', bbox_to_anchor=(0.5, -0.175))
plt.tight_layout()
plt.savefig("exp/pcn_agnews_curves.jpg") ## save plot to disk
plt.clf()