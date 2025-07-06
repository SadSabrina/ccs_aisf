import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, SparsePCA
from sklearn.manifold import TSNE


def plot_pca_or_tsne_layerwise(X_pos, X_neg, hue, standardize=True, reshape= None, n_components=5, components=[0, 1], mode='pca', plot_title=None):
    """
    PCA or T-SNE-clustering for each hidden layer plot

    Parameters:
        X_pos (np.ndarray): Positive (yes) samples, shape (n_samples, n_layers, hidden_dim). Must be normalized after extraction (X_pos - X_pos.mean(0))
        X_neg (np.ndarray): Negative (no) samples, shape (n_samples, n_layers, hidden_dim).  Must be normalized after extraction (X_neg - X_neg.mean(0))
        hue (np.ndarray | pd.Series ): y values 
        standardize (bool):If standardization is needed before PCA (recommended)
        reshape (list): if data has size n_exmaples*n_layers, hidden_dim
        n_components (int): Number of PCA/TSNE components
        components (list): 2 components to plot
        mode (str): 'pca' or 'tsne'
        plot_title (str): figure suptitle

    Note:
      X_pos and X_neg must be normalized after extraction (X_pos - X_pos.mean(0)). 
      This is not important for visualization, but it is important for the correct construction of the CCS.
    """

    if len(X_pos.shape) == 2:
      reshape = [int(i) for i in input('Get reshape params (len data, n_layers)').split(',')]

      X_pos = X_pos.reshape(reshape[0],reshape[1], -1)
      X_neg = X_neg.reshape(reshape[0],reshape[1], -1)

    n_layers = X_pos.shape[1]


    try:
      fig, axes = plt.subplots((n_layers - 1)//6 + 1, 6, figsize=(24, 13))
    except:
      fig, axes = plt.subplots((n_layers - 1)//6 + 1 + 1, 6, figsize=(24, 13))

    axes = axes.flatten()  


    for layer_idx in range(1, n_layers):
        ax = axes[layer_idx-1]

        X_pos_layer = X_pos[:, layer_idx, :]
        X_neg_layer = X_neg[:, layer_idx, :]

        # Combining hidden states of a model to plot a graph
        states_data = X_pos_layer - X_neg_layer


        # Standartization
        if standardize:
            states_data = (states_data - states_data.mean(axis=0)) / states_data.std(axis=0)

        if mode == 'pca':
          projector = PCA(n_components=n_components)  # is good for encoder-decoder
        if mode == 'tsne':
          projector = TSNE(n_components=n_components, metric='cosine') # good for encoder only | decoder only
        if mode == 'pca-sparse':
          projector = SparsePCA(n_components=n_components, alpha=0)

        X_proj = projector.fit_transform(states_data)

        # Plot
        sns.scatterplot(data=pd.DataFrame(X_proj), x=X_proj[:, components[0]], y=X_proj[:, components[1]], hue=hue, ax=ax);
        ax.set_title(f'Layer {layer_idx}', fontsize=10)
        ax.legend().set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True)

  
        for idx in range(n_layers, len(axes)):
          axes[idx].axis('off')

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        try:
          title = hue.name
        except:
          title = None
          
        fig.legend(handles, labels, loc='upper right', fontsize=12, title=title);
        plt.close('all')
    
    fig.suptitle(plot_title, fontsize=16)

    return fig


def get_results_table(ccs_results):

  acc_list = []
  slh_list = []
  agreement_list = []
  agreement_abs_list = []
  ci_list = []
  im_dist_list = []

  for layer in ccs_results.keys():
    acc_list.append(ccs_results[layer]['accuracy'])
    slh_list.append(ccs_results[layer]['silhouette'])
    agreement_list.append(np.mean(ccs_results[layer]['agreement']))
    agreement_abs_list.append(np.median(np.abs(ccs_results[layer]['agreement'])))
    ci_list.append(np.mean(ccs_results[layer]['contradiction idx']))
    im_dist_list.append(np.mean(ccs_results[layer]['IM dist']))

  data = pd.DataFrame(index=ccs_results.keys(),
                      data=np.array([acc_list, agreement_list, agreement_abs_list, ci_list, im_dist_list]).T,
                      columns = ['accuracy', 'agreement_score ↓', 'abs_agreement_score', 'contradiction idx ↓', 'ideal model dist ↓'])
  data['layer_number'] = range(1, len(data)+1)
  data['relative_postion'] = round((data['layer_number'] - 1) /len(data)*100)

  return data