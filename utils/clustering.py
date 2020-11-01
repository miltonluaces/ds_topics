# Silhouette analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

def silhouette_scoring(ds, label_column):
    y = ds[label_column]
    X = ds.drop([label_column], axis=1)
    return silhouette_score(X,y)
  
def silhouette_analysis(ds, label_column):
    y = ds[label_column]
    X = ds.drop([label_column], axis=1)
    levels = pd.unique(y)
    n_levels = len(levels)

    label_idxs = {val : idx + 1 for idx, val in enumerate(levels)} 
    label_levels_idxs = pd.Series([label_idxs[label] for label in y])

    s_avg = silhouette_score(X,y)
    s_values = silhouette_samples(X,y)

    fig, (ax1) = plt.subplots(1)
    fig.set_size_inches(18, 12)

    #ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_levels + 1) * 10])

    y_lower = 10

    for i in range(n_levels):
        i_level_svalues = s_values[label_levels_idxs == i+1]
        i_level_svalues.sort()

        i_size = i_level_svalues.shape[0]
        y_upper = y_lower + i_size

        color = cm.nipy_spectral(float(i) / n_levels)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),  0, i_level_svalues, facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * i_size, levels[i])

        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silhouette analysis")
    ax1.set_xlabel("score")
    ax1.set_ylabel("levels")

    ax1.axvline(x=s_avg, color="red", linestyle="--")

    ax1.set_yticks([])  
    #ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    print('Silhouette score = ', round(s_avg,4))
    plt.show()

def weight_features(ds, label_column, weights):
    dsf = ds.drop(label_column, axis=1)
    wds = dsf * weights
    wds['target'] = ds['target']
    return wds

def weighted_silhouette(ds, label_column, weights):
    wds = weight_features(ds, label_column, weights)
    return silhouette_scoring(wds, 'target')

def get_candidate_weights(step, add_one=True):
    w = np.arange(step,1,step)
    if add_one and (w[len(w)-1]) != 1: w = np.append(w,1)
    return w

def get_candidate_weights_grid(candidate_weights):
    return np.array(np.meshgrid(*candidate_weights)).T.reshape(-1, len(candidate_weights)) 


if __name__ == "__main__":
    print('')