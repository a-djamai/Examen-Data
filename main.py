from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import prince
from sklearn.cluster import KMeans


def dim_red(mat, p, method):
    if method=='ACP':
        if not isinstance(mat, pd.DataFrame):
            try:
                    mat = pd.DataFrame(mat)
            except Exception as e:
                    raise ValueError(f"Impossible de convertir en DataFrame: {e}")
            pca = prince.PCA(n_components=p)
            pca = pca.fit(mat)
            red_mat = pca.transform(mat)

        return red_mat 
        
    elif method=='AFC':
        red_mat = mat[:,:p]
        
    elif method=='UMAP':
        red_mat = mat[:,:p]
        
    else:
        raise Exception("Please select one of the three methods : APC, AFC, UMAP")
    
    return red_mat


def clust(mat, k):
    pred = KMeans(n_clusters=k, n_init="auto").fit(mat)   
    return pred 

def plot_ACP(mat):
    red_mat = dim_red(mat,2)
    pred = clust(red_mat,20)
    centroids = pred.cluster_centers_
    plt.figure(figsize=(8, 6))
    u_labels = np.unique(pred.labels_)
    for i in u_labels:
        # Filtrer les données pour chaque cluster
        cluster_data = red_mat[pred.labels_ == i]
        # Tracer le cluster en utilisant les colonnes spécifiées
        plt.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], label=f'Cluster {i}')

    # Tracer les centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='black', marker='x', label='Centroids')

    plt.title('K-Means Clustering')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(loc='best')
    plt.show()


def cross_validation(red_emb,N):
  nmi = []
  ari = []
  for i in range(N):

    pred,centroids = clust(red_emb, 20)
    nmi_score = normalized_mutual_info_score(pred,labels)
    ari_score = adjusted_rand_score(pred,labels)
    nmi.append(nmi_score)
    ari.append(ari_score)
    
  print(f"NMI scores : {nmi}")
  print(f"ARI scores : {ari}")
  print(f"the mean of NMI is {np.mean(nmi)}")
  print(f"the mean of ARI is {np.mean(ari)}")

# import data
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]
k = len(set(labels))

# embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)

# Perform dimensionality reduction and clustering for each method
methods = ['ACP', 'AFC', 'UMAP']
for method in methods:
    # Perform dimensionality reduction
    red_emb = dim_red(embeddings, 20, method)

    # Perform clustering
    pred = clust(red_emb, k)

    # Evaluate clustering results
    nmi_score = normalized_mutual_info_score(pred, labels)
    ari_score = adjusted_rand_score(pred, labels)

    # Print results
    print(f'Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')

