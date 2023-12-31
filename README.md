# Dimensionality Reduction in Text Clustering

## Overview
This repository demonstrates the use of three popular dimensionality reduction techniques — Principal Component Analysis (PCA), Uniform Manifold Approximation and Projection (UMAP), and t-Distributed Stochastic Neighbor Embedding (t-SNE) — for text clustering tasks, utilizing the 20 Newsgroups dataset from sklearn.datasets.

## Data: 20 Newsgroups Dataset
We employ the 20 Newsgroups dataset, which contains documents classified into 20 distinct topics, making it an ideal resource for experiments in text classification and clustering.

## Getting Started

### Prerequisites
Before starting, ensure you have:
- Python (version 3.8 or later)
- pip (Python package installer)

### Installation

1. **Clone the Repository**:
   ```
   git clone [repository URL]
   cd [repository name]
   ```

2. **Set Up a Virtual Environment** (Optional but recommended):
   - Windows:
     ```
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - macOS and Linux:
     ```
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Required Packages**:
   ```
   pip install -r requirements.txt
   ```
   
### Running the Project
Execute the project with:
```
python main.py
```
Replace `main.py` with your primary script name.

## Docker Image

We have packaged this project into a Docker image for ease of use and deployment. The image includes all necessary dependencies and is preconfigured to run the application.

### Pulling the Docker Image

To use the Docker image, first ensure you have Docker installed on your system. Then, pull the image from Docker Hub using the following command:

```bash
docker pull zaki1003/examen-data
```

### Running the Docker Container

After pulling the image, you can run the container with:

```bash
docker run -p 4000:80 zaki1003/examen-data
```


### Additional Notes
- Ensure the 20 Newsgroups dataset is downloaded and prepared before executing the clustering algorithms.
- Customize configurations (e.g., the number of K-Means clusters) as needed.

## Dimensionality Reduction Techniques
Our approach to managing high-dimensional text data includes:

### PCA
A statistical method that transforms correlated features into linearly uncorrelated components. [Learn more about PCA](link_to_PCA).

### UMAP
An efficient alternative to t-SNE, UMAP is adept at preserving the global structure in large datasets. [Explore UMAP](link_to_UMAP).

### t-SNE
Ideal for visualizing high-dimensional data by mapping similarities into joint probabilities. [Discover t-SNE](link_to_t-SNE).

## Clustering with K-Means
Following dimensionality reduction, we apply the K-Means algorithm for clustering and evaluate its efficacy using:

### NMI (Normalized Mutual Information)
Measures shared information between actual and predicted clusters, scaling Mutual Information (MI) from 0 to 1.

### ARI (Adjusted Rand Index)
An adjustment of the Rand Index, scaling from -1 to 1 to indicate clustering accuracy against a ground truth.


### Results:
#### UMAP:
![umap](https://github.com/a-djamai/Examen-Data/assets/86471508/3732fb13-4d9b-4e4b-b5c1-4a56aa6e91cc)

#### TSNE:
![tsne](https://github.com/a-djamai/Examen-Data/assets/86471508/37228292-74f9-4d7b-bd4a-a3d8308b6df3)

#### PCA:
![acp](https://github.com/a-djamai/Examen-Data/assets/86471508/0504c462-a26f-405e-80b5-bfef99d1a858)

