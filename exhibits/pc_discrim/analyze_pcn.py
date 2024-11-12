from jax import numpy as jnp, random
import numpy as np
import sys, getopt as gopt, time
from pcn_model import PCN as Model  # Custom model import
from ngclearn.utils.metric_utils import measure_ACC, measure_CatNLL
from ngclearn.utils.viz.dim_reduce import extract_tsne_latents, plot_latents

"""
################################################################################
Predictive Coding Network (PCN) Exhibit File:

Evaluates a trained PCN classifier on the test set and produces
a t-SNE visualization of its penultimate layer activities.

Usage:
$ python analyze_pcn.py --dataX="/path/to/test_patterns.npy" \
                        --dataY="/path/to/test_labels.npy"

@author: The Neural Adaptive Computing Laboratory
################################################################################
"""

# Function to evaluate model's performance
def eval_model(model, Xdev, Ydev, mb_size):
    n_batches = int(Xdev.shape[0] / mb_size)
    latents = []
    nll = 0.0  # negative Categorical log-likelihood
    acc = 0.0  # accuracy
    n_samp_seen = 0

    for j in range(n_batches):
        idx = j * mb_size
        Xb = Xdev[idx: idx + mb_size, :]
        Yb = Ydev[idx: idx + mb_size, :]

        # Run model inference
        yMu_0, yMu, _ = model.process(obs=Xb, lab=Yb, adapt_synapses=False)
        latents.append(model.get_latents())

        # Record metric measurements
        _nll = measure_CatNLL(yMu_0, Yb) * Xb.shape[0]  # un-normalized score
        _acc = measure_ACC(yMu_0, Yb) * Yb.shape[0]  # un-normalized score
        nll += _nll
        acc += _acc
        n_samp_seen += Yb.shape[0]

    # Normalize scores
    nll /= Xdev.shape[0]
    acc /= Xdev.shape[0]
    latents = jnp.concatenate(latents, axis=0)
    return nll, acc, latents

# Read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '', ["dataX=", "dataY="])

dataX = "../../data/agnews/testX.npy"
dataY = "../../data/agnews/testY.npy"
for opt, arg in options:
    if opt == "--dataX":
        dataX = arg.strip()
    elif opt == "--dataY":
        dataY = arg.strip()

print(f"=> Data X: {dataX} | Y: {dataY}")

# Load data
_X = jnp.load(dataX)
_Y = jnp.load(dataY)

# Ensure data and labels have the same length
if _X.shape[0] != _Y.shape[0]:
    print(f"Warning: Length of _X ({_X.shape[0]}) and _Y ({_Y.shape[0]}) do not match.")
    _Y = _Y[:_X.shape[0]]  # Truncate _Y to match _X if necessary

dkey = random.PRNGKey(time.time_ns())
model = Model(dkey=dkey, loadDir="exp/pcn")  # Load pre-trained PCN model

# Evaluate performance
nll, acc, latents = eval_model(model, _X, _Y, mb_size=1000)
print("------------------------------------")
print(f"=> NLL = {nll}  Acc = {acc}")

# Extract latents and visualize with t-SNE
print("Lat.shape =", latents.shape)
codes = extract_tsne_latents(np.asarray(latents))
print("code.shape =", codes.shape)

# Plotting the 2D latent encodings
if codes.shape[0] != _Y.shape[0]:
    print(f"Warning: Length of codes ({codes.shape[0]}) and _Y ({_Y.shape[0]}) do not match. Truncating labels.")
    plot_latents(codes, _Y[:codes.shape[0]], plot_fname="exp/pcn_latents.jpg")
else:
    plot_latents(codes, _Y, plot_fname="exp/pcn_latents.jpg")
