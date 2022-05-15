import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors

def run_tsne(data, n_examples, experts_list, out_pic_name):
    # data.shape: bs * dim
    tsne = TSNE(n_components=2, init='pca', random_state=2)
    tsne_result = tsne.fit_transform(data)
    color_list = list(pltcolors.get_named_colors_mapping().keys())
    random.shuffle(color_list)
    for i, coord in enumerate(tsne_result):
        if i < n_examples * len(experts_list):
            idx = i // n_examples
            if i % n_examples == 0:
                plt.scatter(coord[0], coord[1], marker='x', c=color_list[idx], label=experts_list[idx])
            else:
                plt.scatter(coord[0], coord[1], marker='x', c=color_list[idx])
        elif i >= n_examples * len(experts_list):
            idx += 1
            print(coord)
            plt.scatter(coord[0], coord[1], marker='o', c=color_list[i-n_examples*len(experts_list)], label=experts_list[idx-len(experts_list)])
        
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.title(f't-SNE of \'{experts_list}\' Samples & PreferenceVector')
    plt.savefig(out_pic_name)
