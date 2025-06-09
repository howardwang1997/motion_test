import numpy as np
import pandas as pd

from data import read_data, make_dataset
from clustering import clustering_tsne, clustering_pca
from draw import scatter_tsne


def main():
	window = 50
	
	path = '../../data/snowboard/20250524/S_turn_1.txt'
	data = read_data(path)
	assert data.isnull().values.any() == 0
	dataset_1 = make_dataset(data, window=window)
	
	path = '../../data/snowboard/20250524/S_turn_2.txt'
	data = read_data(path)
	assert data.isnull().values.any() == 0
	dataset_2 = make_dataset(data, window=window)
	
	path = '../../data/snowboard/20250524/straight.txt'
	data = read_data(path)
	assert data.isnull().values.any() == 0
	dataset_3 = make_dataset(data, window=window)
	
	path = '../../data/snowboard/20250524/J_front.txt'
	data = read_data(path)
	assert data.isnull().values.any() == 0
	dataset_4 = make_dataset(data, window=window)
	
	path = '../../data/snowboard/20250524/J_back.txt'
	data = read_data(path)
	assert data.isnull().values.any() == 0
	dataset_5 = make_dataset(data, window=window)
	
	dataset = np.vstack([dataset_1, dataset_2, dataset_3, dataset_4, dataset_5])
	
	# print(dataset)
	print(dataset_1.shape, dataset_2.shape, dataset_3.shape, dataset_4.shape, dataset_5.shape)
	lengths = [dataset_1.shape[0] + dataset_2.shape[0], dataset_3.shape[0], dataset_4.shape[0], dataset_5.shape[0]]
	seps = [0]
	for l in lengths:
		seps.append(seps[-1] + l)
	labels = ['S turns', 'straight run', 'J turn front', 'J turn back']
	colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']
	
	embed = clustering_tsne(dataset)
	print(seps, lengths)
	scatter_tsne(embed, seps, labels, colors, save_path=f'../../data/snowboard/20250524/TSNE_{window}.png')
	
if __name__ == '__main__':
	main()
