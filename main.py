from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import prince
from sklearn.cluster import KMeans
import umap
import umap.plot
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt




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
        
    elif method=='TSNE':
        red_mat = TSNE(n_components=p, learning_rate='auto',init='random').fit_transform(mat)
                
    elif method=='UMAP':
        category_labels = [ng20.target_names[x] for x in ng20.target]
        hover_df = pd.DataFrame(category_labels, columns=['category'])
        hover_df = hover_df[:2000]

        reducer = umap.UMAP(n_components=p)
        red_mat = reducer.fit_transform(mat)

        mapper = umap.UMAP().fit(mat)
        umap.plot.points(mapper, labels=hover_df['category'])
        
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



#plotting the results:
def plot_TSNE(mat):
    # perform dimentionality reduction
    pred_final = dim_red(mat, 2)
    pred_clust = clust(pred_final, k)

    pred_labels= pred_clust.labels_
    #Getting the Centroids
    centroids = pred_clust.cluster_centers_
    u_labels = np.unique(pred_labels)

    for i in u_labels:
        plt.scatter(pred_tsne[pred_labels == i , 0] , pred_tsne[pred_labels == i , 1] , label = i)
    plt.title('K-means Clustering (TSNE)')
    plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
    plt.legend()
    plt.show()


    
def cross_validation(mat,N):
  nmi = []
  ari = []
  for i in range(N):
    pred = clust(mat, 20)
    nmi_score = normalized_mutual_info_score(pred.labels_,labels)
    ari_score = adjusted_rand_score(pred.labels_,labels)
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
methods = ['ACP', 'TSNE', 'UMAP']
for method in methods:
    n_components = 20 

    if(method == 'TSNE'):
        n_components=3

    # Perform dimensionality reduction
    red_emb = dim_red(embeddings, n_components, method)

    # Perform clustering
    pred = clust(red_emb, k)

    # Evaluate clustering results
    nmi_score = normalized_mutual_info_score(pred, labels)
    ari_score = adjusted_rand_score(pred, labels)

    if(method == 'TSNE'):
        # perform dimentionality reduction
        plot_TSNE(embeddings)
    print("Cross Validation for ", method)   
    cross_validation(red_emb,10)
    # Print results
    print(f'Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')
