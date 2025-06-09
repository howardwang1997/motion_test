from matplotlib import pyplot as plt

def scatter_tsne(data_embed, seps, labels, colors, save_path='TSNE.png'):
	data_embed = data_embed.transpose()
	
	fig, ax = plt.subplots()
	for i in range(len(seps) - 1):
		ax.scatter(data_embed[0, seps[i]:seps[i+1]], data_embed[1, seps[i]:seps[i+1]],
				   s=2, color=colors[i], label=labels[i])
	# ax.scatter(data_embed[0, 327:490], data_embed[1, 327:490], s=2, color='orange', label='straight run')
	# ax.scatter(data_embed[0, 490:607], data_embed[1, 490:607], s=2, color='red', label='J turn front')
	# ax.scatter(data_embed[0, 607:], data_embed[1, 607:], s=2, color='green', label='J turn back')
	
	ax.set_title('Visualization of Snowboarding')
	ax.legend()
	fig.show()
	fig.savefig(save_path)
