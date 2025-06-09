import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from data import read_data, make_dataset


def clustering_tsne(data: np.array): #-> np.array
	tsne = TSNE(n_components=2, init='pca')
	data_embed = tsne.fit_transform(data)
	return data_embed


def clustering_pca(data: np.array): #-> np.array
	pca = PCA(n_components=2)
	data_embed = pca.fit_transform(data)
	return data_embed
