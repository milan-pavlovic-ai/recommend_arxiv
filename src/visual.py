"""Visualization methods"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import networkx as nx

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import BoundaryNorm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from data import DataManager


class PlotManager:
    """Visualization methods"""
    
    COLORS = ['blue', 'green', 'red']
    
    def __init__(self, datamngr: DataManager) -> None:
        """Constructor

        Args:
            datamngr (DataManager): Manager of data
            
        Returns:
            None
        """
        self.datamngr = datamngr
        self.net = self.datamngr.net
        self.embds = self.datamngr.embds
        self.viz_name = self.datamngr.file_data.split('_')[1].split('.')[0]
        return
    
    def plot_network(self):
        """Plot network"""
        # Export graph
        nx.write_gexf(self.net, f'data/gephi_{self.viz_name}.gexf')
        
        # Plot
        plt.figure(num=None, figsize=(20, 20), dpi=80)
        plt.axis('off')
        pos = nx.spring_layout(self.net)
        nx.draw_networkx_nodes(self.net, pos)
        #nx.draw_networkx_edges(self.net, pos)
        nx.draw_networkx_labels(self.net, pos)

        cut = 1.00
        xmax = cut * max(xx for xx, yy in pos.values())
        ymax = cut * max(yy for xx, yy in pos.values())
        
        xmin = cut * min(xx for xx, yy in pos.values())
        ymin = cut * min(yy for xx, yy in pos.values())
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        plt.savefig(f'data/graph_{self.viz_name}1.png', bbox_inches='tight')
        plt.show()
        plt.close()
        return
    
    def plot_data(self, num_dim=3, use_tsne=False) -> None:
        """Visualize high-dimensional data into 2D and 3D space with PCA technique

        Args:
            num_dim (int, optional): Number of dimensions. Defaults to 3.
            use_tsne (bool, optional): Use t-SNE technique. Defaults to False.

        Raises:
            ValueError: Invalid parameters
        """
        # Check dimensions and format
        if num_dim < 1:
            raise ValueError('Invalid number of dimension')

        # Standardize
        X_data = StandardScaler().fit_transform(self.embds)
        X_data = pd.DataFrame(X_data, columns=self.embds.columns)

        # PCA
        pca = PCA(n_components=num_dim)
        X_data_reduced = pca.fit_transform(X_data)
        d0 = X_data_reduced[:, 0]
        d1 = np.zeros_like(d0) if num_dim <= 1 else X_data_reduced[:, 1]
        d2 = np.zeros_like(d0) if num_dim <= 2 else X_data_reduced[:, 2]

        print('Percentage of variance explained by each of the selected components:\n', pca.explained_variance_ratio_)

        # Plot
        PlotManager.__plot_3d_data(d0, d1, d2)

        # t-SNE
        if use_tsne:
            tsne_limit = 50
    
            # Reduce number of feature with PCA to reduce t-SNE computation
            if X_data.shape[1] > tsne_limit:
                pca = PCA(n_components=tsne_limit)
                X_data = pca.fit_transform(X_data)

            # Create t-SNE dimensions
            tsne = TSNE(n_components=num_dim, n_jobs=-1, n_iter=500)
            X_data_reduced = tsne.fit_transform(X_data)

            d0 = X_data_reduced[:, 0]
            d1 = np.zeros_like(d0) if num_dim <= 1 else X_data_reduced[:, 1]
            d2 = np.zeros_like(d0) if num_dim <= 2 else X_data_reduced[:, 2]

            # Plot
            PlotManager.__plot_3d_data(d0, d1, d2)
        return

    @staticmethod
    def __plot_3d_data(d0, d1, d2, y_data=None):
        """
        Plot data in 3D space
        """
        if y_data is not None:
            if num_classes is None:
                num_classes = len(np.unique(y_data))
            colors = list(PlotManager.COLORS[:num_classes])
            cmap = mpl.colors.ListedColormap(colors)                    # Define the colormap
            min_val = y_data.min()
            max_val = y_data.max() + 1
            bounds = np.linspace(min_val, max_val, num_classes + 1)     # Define the bins and normalize
            norm = BoundaryNorm(bounds, cmap.N)

        fig, axes = plt.subplots(figsize=(10, 8), dpi=200)
        #axes = fig.add_axes([0.85, 0.16, 0.02, 0.65])  # left, bottom, right, top
        axes.get_xticklabels([])
        axes.get_yticklabels([])
        plt.axis('off')
        ax = Axes3D(fig, elev=-150, azim=110, title='3D Chart')
        
        if y_data is not None and num_classes > 1:
            scatter_plot = ax.scatter(d0, d1, d2, c=y_data, cmap=cmap, edgecolor='k', norm=norm, s=40, alpha=0.6)
            #cbar = mpl.colorbar.ColorbarBase(axes, cmap=cmap, norm=norm, ticks=bounds, boundaries=bounds, orientation='vertical', drawedges=False)
            cbar = plt.colorbar(scatter_plot, spacing='proportional', ticks=bounds, ax=axes)
            labels = np.arange(min_val, max_val, 1)
            cbar.set_ticks(labels + 0.5)
            #cbar.set_ticklabels(['Normal', 'Anomaly'])
            cbar.set_ticklabels(labels)
            # cbar.set_label('Device ID')
        else:
            #scatter_plot = ax.scatter(d0, d1, d2, edgecolor='k', s=40, alpha=0.6)
            scatter_plot = ax.scatter(d0, d1, d2, edgecolor='k', color='blue', s=40, alpha=0.6)
    
        ax.set_title('3D Chart')
        ax.set_xlabel('D1')
        ax.set_ylabel('D2')
        ax.set_zlabel('D3')
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        
        plt.show()
        plt.close()
        return


if __name__ == '__main__':
    
    # Init data
    datamngr = DataManager(file_data='data/arxiv_3x100.csv')
    datamngr.data()

    # Init network
    datamngr.network(node='authors', links=['id'], compact=1)

    # Init Embeddings
    datamngr.embeddings()
    
    # Visualizaitons
    pltmngr = PlotManager(datamngr)
    pltmngr.plot_network()
    pltmngr.plot_data()
    
    pass
