"""Managment of data"""
import os
import arxiv
import random
import numpy as np
import pandas as pd
import networkx as nx

from tqdm import tqdm
from ast import literal_eval
from node2vec import Node2Vec
from itertools import product
from sklearn.model_selection import train_test_split


class DataManager:
    """Managment of data"""
     
    MAX_COLUMNS = 10
    RANDOM_SEED = 21
     
    def __init__(self,
                file_data: str,
                max_papers: int = 100,
                queries: list = None
        ) -> None:
        """Initialize data manager.

        Args:
            file_data (str):  Location of data file.
            max_papers (int, optional): Maximum number of papers. Defaults to 100.
            queries (list, optional): List of topics for search. Defaults to None.
            
        Returns
            None
        """
        self.file_data = file_data
        self.max_papers = max_papers
        self.queries = queries
        
        self.df = None          # Original DF
        self.df_norm = None     # Normalized DF without list-like elements
        self.net = None         # Network created from normalized DF
        self.embds = None       # Network embeddings as DF
        self.file_embds = None  # Name of file with embeddings
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None 
        return
    
    def __read_data(self, file_data: str) -> pd.DataFrame:
        """Read data from CSV file.

        Args:
            file_data (str): Location of data file.

        Returns:
            pd.DataFrame: Dataset
        """
        return pd.read_csv(file_data, header=0)
    
    def __find_arxiv(self, 
                     queries: list, 
                     max_papers: int = 100, 
                     file_data: str = True
        ) -> pd.DataFrame:
        """Find arXiv papers based on the given queries.

        Args:
            queries (list): List of topics.
            max_papers (int, optional): Maximum number of papers. Defaults to 100.
            file_data (str, optional): Location where DF will be saved.

        Returns:
            pd.DataFrame: Dataset
        """
        all_results = []
        
        # Get papers from arXiv
        print('\nStart searching on arXiv ...')
        try:
            for query in tqdm(queries):
                results = arxiv.Search(
                    query=query,
                    max_results=max_papers,
                    sort_by = arxiv.SortCriterion.SubmittedDate,
                    sort_order = arxiv.SortOrder.Descending
                )
                all_results.append(results)
            print('\nSearch has been completed.\n')
                
        except Exception as e:
            print('\nUnable to complete arXiv API call')
            print(e)
            exit(0)
            
        # Create DataFrame
        print('Start extracting papers from results ...')
        data = []
        for result in all_results:
            # Extract papers for each search result
            for paper in tqdm(result.results()):
                paper_info = {
                    'title' : paper.title,
                    'published' : paper.published,
                    'id' : paper.entry_id,
                    'url' : paper.pdf_url,
                    'main_topic' : paper.primary_category,
                    'all_topics' : paper.categories,
                    'authors' : paper.authors
                }
                data.append(paper_info)
            
        self.df = pd.DataFrame(data)

        # Remove duplicates
        self.df = self.df.drop_duplicates(subset='title', keep='first')

        # Add year
        print('\nStart pre-processing data ...')
        self.df['year'] = pd.DatetimeIndex(self.df['published']).year.astype(np.int64)

        # Simplfy ids
        ids = self.df['id'].unique()
        papers_dict = {id : i for i, id in enumerate(ids)}
        self.df['id'] = self.df['id'].map(papers_dict)
        
        # Clean authors
        self.df['authors'] = self.df['authors'].apply(lambda x: [str(a).upper() for a in x])
        
        # Separate links
        self.df['url'] = self.df['url'].apply(lambda x: '"{}"'.format(x))
              
        # Save data
        if file_data is not None:
            self.df.to_csv(file_data, index=False)
            print('\nData has been successfuly saved!')
        
        return self.df

    def data(self) -> pd.DataFrame:
        """Get data from cache or arXiv server.

        Returns:
            pd.DataFrame: Dataset
        """
        # Get data
        if not os.path.exists(self.file_data):
            self.__find_arxiv(queries=self.queries, max_papers=self.max_papers, file_data=self.file_data)
        self.df = self.__read_data(self.file_data)
        
        # Show
        with pd.option_context("display.max_columns", DataManager.MAX_COLUMNS):
            print('\nData:')
            print(self.df)
        
        return self.df
    
    def network(self, node: str, links: list, compact: int = -1) -> nx.Graph:
        """Create network based on the df with adjacency list structure.

        Args:
            node (str): Column in df that represents node
            links (list): List of columns in df that represents links
            compact (int): Make compact network by removing all nodes with degree <= compact value. Default to -1.
            
        Returns:
            nx.Graph: Created network from data frame
        """
        # Normalize data
        self.df_norm = self.df.copy()
        
        for link in links + [node]:
            if link not in ('year', 'id'):
                self.df_norm[link] = self.df_norm[link].apply(literal_eval)
                self.df_norm = self.df_norm.explode(link)

        with pd.option_context("display.max_columns", DataManager.MAX_COLUMNS):
            print('\n\nNormalized data:')
            print(self.df_norm)
            
        # Create adjacency table
        adj_dict = {}

        for id, node_group in self.df_norm.groupby(node):
            
            all_link_nodes = []
            for link in links:
                link_values = node_group[link].unique()
                link_df = self.df_norm[(self.df_norm[node] != id) & (self.df_norm[link].isin(link_values))]
                link_nodes = list(link_df[node].unique())
                all_link_nodes += link_nodes
                
            adj_dict[id] = all_link_nodes
        
        # Create undirected multi-graph
        self.net = nx.Graph(adj_dict, create_using=nx.MultiGraph)
        
        # Make compact
        print('\n\nNetwork:')
        if compact != -1:
            to_remove = [n for n, degree in self.net.degree() if degree <= compact]
            self.net.remove_nodes_from(to_remove)
            print(f'Removed nodes {len(to_remove)} with degree <= {compact}')
        
        # Info
        print(f'# of nodes: {self.net.number_of_nodes()}')
        print(f'# of links: {self.net.number_of_edges()}')
        print(f'Avg degree: {sum(dict(self.net.degree()).values()) / self.net.number_of_nodes():.2f}')

        return self.net

    def embeddings(self,
                   dims: int = 128,
                   walk_length: int = 80,
                   num_walks: int = 10,
                   window: int = 10,
                   min_count: int = 1,
                   batch_words: int = 4,
                   n_jobs: int = 16,
                   to_save: bool = True
        ) -> pd.DataFrame:
        """Create embeddings based on Node2Vec method.

        Args:
            dims (int, optional): Embedding dimensions. Defaults to 128.
            walk_length (int, optional): Number of nodes in each walk. Defaults to 80.
            num_walks (int, optional): Number of walks per node. Defaults to 10.
            window (int, optional): Window size. Defaults to 10.
            min_count (int, optional): Minimum count. Defaults to 1.
            batch_words (int, optional): Batch of words. Defaults to 4.
            n_jobs (int, optional): Number of threads. Defaults to 16.

        Returns:
            pd.DataFrame: Node2Vec Embeddings
        """
        # Get embeddings
        suffix = self.file_data.split('_')[1].split('.')[0]
        self.file_embds = f'data/embeddings_{dims}d_arxiv{suffix}.csv'
        
        if os.path.exists(self.file_embds):
            self.embds = self.__read_data(self.file_embds)
            self.embds = self.embds.set_index('index')
            return self.embds
        
        # Precompute probabilities and generate walks
        node2vec = Node2Vec(
            graph=self.net,
            dimensions=dims,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=n_jobs
        )
        
        # Model for embeddings, any keywords acceptable by gensim.Word2Vec
        model = node2vec.fit(
            window=window,
            min_count=min_count,
            batch_words=batch_words
        )
        
        # Create graph embeddings DF
        self.embds = pd.DataFrame(
            data=[model.wv.get_vector(str(id)) for id in self.net.nodes()],
            index=self.net.nodes
        )
        self.embds.index.name = 'index'

        # Show
        with pd.option_context("display.max_columns", DataManager.MAX_COLUMNS):
            print('\n\nEmbeddings data:')
            print(self.embds)
        
        # Save
        if to_save:
            self.embds.to_csv(self.file_embds, index=True)
            print('\nEmbeddings has been sucessfuly saved!')
        
        return self.embds

    def train_test_data(self, test_size: int = 0.3, sample_size: float = 0.1) -> None:
        """Create train-test data

        Args:
            test_size (int, optional): Size of test set. Defaults to 0.3.
            sample_size (float, optional): Portion of edge sampling. Defaults to 0.1.

        Returns:
            None
        """
        nodes = list(self.net.nodes())
        edges = list(self.net.edges())
        
        # All edges
        nodes_pairs = product(nodes, nodes)
        all_edges = [(v, w) for (v, w) in nodes_pairs if v != w]
        
        # Subset of edges
        num_edges = int(len(all_edges) * sample_size)
        sampled_edges = random.sample(all_edges, num_edges)
        subset_edges = sampled_edges + edges

        # Edge features on subset
        edge_features = [
            (self.embds.loc[str(v)] + self.embds.loc[str(w)])
            for v, w in subset_edges
        ]
        X = np.array(edge_features)

        # Targets
        y = np.array([1 if e in edges else 0 for e in subset_edges])
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, stratify=y, test_size=test_size, random_state=DataManager.RANDOM_SEED)
        
        return


if __name__ == '__main__':
    
    # Initialize
    datamngr = DataManager(
        file_data='data/arxiv_3x100.csv',
        max_papers=100,
        queries=[
            'machine learning',
            'recommendation system',
            'link prediction',
        ]
    )
    
    # Get data
    data = datamngr.data()

    # Create network
    net = datamngr.network(
        node='authors',
        links=['id'],
        compact=0
    )

    # Create embeddings
    embds = datamngr.embeddings(
        dims=128,
        walk_length=100,
        num_walks=20,
        window=1,
        min_count=1,
        batch_words=4,
        n_jobs=1,
        to_save=True
    )

    pass
