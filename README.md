# recommend_arxiv
## ArXiv co-author recommendation system based on link predictions

The recommendation system for a given author recommends co-authors with whom collaboration has not been achieved so far. The prediction is based on the connections of authors in arXiv's scientific papers.

### Commands

Create a dataset based on the parameters inside the data.py file.
The parameters include querying papers from the arXiv server, defining nodes and links within the network, as well as creating graph-based embeddings with Node2Vec method.

    make data

By running the following command, the network will be drawn and saved as .png. Also, the network will be exported to the .gexf file that can be loaded using Gephi, a visualization and exploration platform for all kinds of graphs and networks. In addition, graph-based embeddings will be drawn using PCA and/or t-SNE reduction techniques.

    make visual

Generate recommendations with techniques based on Cosine similarity and Random Forest classifier. In addition, within the file recommend.py it is possible to adjust parameters.

    make recommend

With the Poetry dependency management tool, it is possible to easily install all dependecies with the following command:

    make install

The list of necessary libraries for setting up the project:

    python = "^3.8"
    arxiv = "^1.4.2"
    numpy = "^1.23.3"
    pandas = "^1.5.0"
    scikit-learn = "^1.1.2"
    node2vec = "^0.4.6"
    seaborn = "^0.12.0"
    tqdm = "^4.64.1"
    networkx = "2.6.3"


