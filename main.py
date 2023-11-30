from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
import numpy as np
import umap
from sklearn.cluster import KMeans
import umap.plot
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from sklearn.preprocessing import MaxAbsScaler

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def dim_red(mat, p, method):
    '''
    Perform dimensionality reduction

    Input:
    -----
        mat : NxM list 
        p : number of dimensions to keep 
    Output:
    ------
        red_mat : NxP list such that p<<m
    '''
    if method=='ACP':
        red_mat = mat[:,:p]
        
    elif method=='AFC':
        red_mat = mat[:,:p]
        
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
    '''
    Perform clustering

    Input:
    -----
        mat : input list 
        k : number of cluster
    Output:
    ------
        pred : list of predicted labels
    '''
    
    pred = np.random.randint(k, size=len(corpus))
    
    return pred

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

def cross_validate_reduction(red_emb, X, y, n_components=2):
    reduced_X = red_emb
    clf = RandomForestClassifier()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, reduced_X, y, cv=cv, scoring='accuracy')
    return scores

# reduce_with_X = red_emb with method X
umap_scores = cross_validate_reduction(red_emb, corpus, labels)
#pca_scores = cross_validate_reduction(red_emb, corpus, labels)
#tsne_scores = cross_validate_reduction(red_emb, corpus, labels)

print("UMAP Scores:", umap_scores)
#print("PCA Scores:", pca_scores)
#print("t-SNE Scores:", tsne_scores)

