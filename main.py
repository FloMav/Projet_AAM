import pandas as pd

df = pd.read_csv("Data/DATA_PROJECT.csv", index_col=0)

corrMatrix = df.corr()
print(corrMatrix)


# 1. Hierarchical clustering
# ✓ PCA decomposition of the covariance matrix. ✓ Features = ARP exposures to PCs
# ✓ Dendrogram
# - Ward linkage
# - Euclidean distance

# pca = PCA()
# x_train = pca.fit_transform(x_train)
# x_val = pca.transform(x_val)
# explained_variance = pca.explained_variance_ratio_
# res_PCA = pd.DataFrame({'Expl. variance' : np.round(explained_variance,2)}, index = ['PCA' + str(i) for i in range(9)])
# res_PCA
# plt.plot(res_PCA)