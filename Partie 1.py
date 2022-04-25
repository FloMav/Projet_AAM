import pandas as pd
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cut_tree
from matplotlib import pyplot as plt

df = pd.read_csv("Data/DATA_PROJECT.csv", index_col=0)
corrMatrix = df.corr()

returns = df.drop(df.columns[-1], axis=1)
label = returns.columns
returns = preprocessing.scale(returns)

dist = pdist(returns.T, 'correlation')
output = linkage(dist, method='ward')

plt.figure(figsize=(10, 10))
csfont = {'fontname': 'Calibri', 'fontsize': '10'}
csfont2 = {'fontname': 'Calibri', 'fontsize': '8'}
csfont3 = {'fontname': 'Calibri', 'fontsize': '6'}
plt.xlabel('Distance', **csfont2)
plt.xticks(**csfont3)
plt.yticks(**csfont3)
dendrogram(output, color_threshold=1.5, truncate_mode='level', orientation='right', leaf_font_size=10, labels=label)
plt.savefig('Data/books_read.jpeg', bbox_inches='tight', dpi=300)
plt.show()

clusters = fcluster(output, t=1.3, criterion='distance')

cluster_1 = []
cluster_2 = []

for i in range(len(clusters)):
    if clusters[i] == 1:
        cluster_1.append(label[i])
    else:
        cluster_2.append(label[i])

df1 = pd.DataFrame(index=label)
df1['clusters'] = clusters
