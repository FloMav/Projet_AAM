import pandas as pd
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cut_tree
from matplotlib import pyplot as plt

df = pd.read_csv("Data/DATA_PROJECT.csv", index_col=0, sep=';')

corrMatrix = df.corr()
print(corrMatrix)

# 1. Hierarchical clustering
# ✓ PCA decomposition of the covariance matrix. ✓ Features = ARP exposures to PCs
# ✓ Dendrogram
# - Ward linkage
# - Euclidean distance

# pca = PCA() x_train = pca.fit_transform(x_train) x_val = pca.transform(x_val) explained_variance =
# pca.explained_variance_ratio_ res_PCA = pd.DataFrame({'Expl. variance' : np.round(explained_variance,2)},
# index = ['PCA' + str(i) for i in range(9)]) res_PCA plt.plot(res_PCA)

returns = preprocessing.scale(df)

dist = pdist(returns.T, 'correlation')
output = linkage(dist, method='ward')

# GRAPH
plt.figure(figsize=(10, 10))
csfont = {'fontname': 'Calibri', 'fontsize': '10'}
csfont2 = {'fontname': 'Calibri', 'fontsize': '8'}
csfont3 = {'fontname': 'Calibri', 'fontsize': '6'}
plt.xlabel('Distance', **csfont2)
plt.xticks(**csfont3)
plt.yticks(**csfont3)
dendrogram(output, color_threshold=1.5, truncate_mode='level', orientation='right', leaf_font_size=10, labels=LABl)
plt.savefig('books_read.jpeg', bbox_inches='tight', dpi=300)
plt.show()
