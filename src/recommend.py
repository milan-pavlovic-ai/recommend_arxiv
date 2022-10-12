"""Recommendation system based on link predictions"""

import numpy as np

from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, f1_score, roc_auc_score, matthews_corrcoef

from data import DataManager
from visual import PlotManager


class Recommender:
    """Recommendation system based on link predictions"""
    
    def __init__(self, file_data: str, test_size: float, sample_size: float) -> None:
        """Constructor

        Args:
            file_data (str): Location of data file.
            test_size (float): Percentage of data for testing.
            sample_size (float): Percentage of edges.
            
        Returns:
            None
        """
        # Data
        self.datamngr = DataManager(file_data=file_data)
        self.datamngr.data()

        # Network
        self.net = self.datamngr.network(node='authors', links=['id'], compact=0)

        # Embeddings
        self.embds = self.datamngr.embeddings()
        
        # Visualization
        self.plotmngr = PlotManager(self.datamngr)
        
        # Model
        self.model = None
        
        # Data for model
        self.datamngr.train_test_data(test_size=test_size, sample_size=sample_size)
        self.X_train = self.datamngr.X_train
        self.X_test = self.datamngr.X_test
        self.y_train = self.datamngr.y_train
        self.y_test = self.datamngr.y_test
        return
    
    def recommend_with_similarity(self, id: str, top_k: int = 5) -> list:
        """Recommend with similarity-based method.

        Args:
            id (str): Based on this node apply recommendation
            top_k (int, optional): Top K recommendations. Defaults to 5.

        Returns:
            list: List of recommendations
        """
        # Check parameters
        id = id.upper()
        if id not in self.embds.index:
            print(f'Node: {id} does not exist!')
            return []
        
        # Node of interest
        node_focus = self.embds[self.embds.index == id]
        
        # Candidates
        nodes = self.net.nodes()
        neighbors = list(self.net.adj[id]) + [id]
        candidate_nodes = [n for n in nodes if n not in neighbors]
        candidates = self.embds[self.embds.index.isin(candidate_nodes)]
        
        # Calculate predictions
        preds = cosine_similarity(node_focus, candidates)[0].tolist()
        
        # Similarity dictionary
        indices = candidates.index.tolist()
        preds_dict = dict(zip(indices, preds))
        preds_dict_sort = sorted(preds_dict.items(), key=lambda x: x[1], reverse=True)
        
        recommends = preds_dict_sort[:top_k]
        return recommends

    def train_model(self):
        # Train model
        self.model = RandomForestClassifier(
            n_jobs=-1,
            random_state=DataManager.RANDOM_SEED
        )
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate model
        y_preds = self.model.predict(self.X_train)
        print('\nTRAIN results:')
        print(f'F score: {f1_score(self.y_train, y_preds)*100:.2f}')
        print(f'ROC AUC: {roc_auc_score(self.y_train, y_preds)*100:.2f}')
        print(f'MCC score: {matthews_corrcoef(self.y_train, y_preds):.2f}')
        print(classification_report(self.y_train, y_preds))
        
        y_preds = self.model.predict(self.X_test)
        print('\nTEST results:')
        print(f'F score: {f1_score(self.y_test, y_preds)*100:.2f}')
        print(f'ROC AUC: {roc_auc_score(self.y_test, y_preds)*100:.2f}')
        print(f'MCC score: {matthews_corrcoef(self.y_test, y_preds):.2f}')
        print(classification_report(self.y_test, y_preds))
        return

    def recommend_with_model(self, id: str, top_k: int = 5) -> list:
        """Recommend with ML-based approach

        Args:
            id (str): Based on this node apply recommendation
            top_k (int, optional): Top K recommendations. Defaults to 5.

        Returns:
            list: Top K recommendations
        """
        # Check parameters
        id = id.upper()
        if id not in self.embds.index:
            print(f'Node: {id} does not exist!')
            return []
        
        # Edges candidates
        nodes = list(self.net.nodes())
        neighbors = [id] + list(self.net.adj[id])
        
        candidate_nodes = [n for n in nodes if n not in neighbors]
        all_edges = product([id], candidate_nodes)
        
        edge_features = np.array([
            (self.embds.loc[str(v)] + self.embds.loc[str(w)])
            for v, w in all_edges
        ])

        # Predictions
        preds = self.model.predict_proba(edge_features)
        preds = preds.max(axis=1)
        
        preds_dict = dict(zip(candidate_nodes, preds))
        preds_dict_sort = sorted(preds_dict.items(), key=lambda x: x[1], reverse=True)
        
        recommends = preds_dict_sort[:top_k]
        return recommends


if __name__ == '__main__':
    
    # Initialize
    recsys = Recommender(
        file_data='data/arxiv_3x100.csv',
        test_size=0.3,
        sample_size=0.015
    )
    
    
    # Recommend with Similarity
    results = recsys.recommend_with_similarity(
        id='DANIELA RUS',
        top_k=3
    )
    print(f'\nResults with Cosine similarity:')
    for item in results:
        print(item)
    
    results = recsys.recommend_with_similarity(
        id='PHILIPP NEUBAUER',
        top_k=3
    )
    print(f'\nResults with Cosine similarity:')
    for item in results:
        print(item)


    # Recommend with Model
    recsys.train_model()
    
    results = recsys.recommend_with_model(
        id='DANIELA RUS',
        top_k=3
    )
    print(f'\nResults with RF model:')
    for item in results:
        print(item)
        
    results = recsys.recommend_with_model(
        id='PHILIPP NEUBAUER',
        top_k=3
    )
    print(f'\nResults with RF model:')
    for item in results:
        print(item)

    pass
