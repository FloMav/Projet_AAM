import pandas as pd
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cut_tree
from matplotlib import pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats



df = pd.read_csv("Data/DATA_PROJECT.csv", index_col=0)
corrMatrix = df.corr()

market = df.iloc[:, -1]
returns = df.drop(df.columns[-1], axis=1)


def clusterisation(returns):
    returns=preprocessing.scale(returns, axis=1)
    dist = pdist(returns.T, 'correlation')
    output = linkage(dist, method='ward')

    n_clust=2
    clusters = fcluster(output, n_clust,criterion='maxclust')

    return  clusters

def dataframe_clusters(returns):
    df_clust=pd.DataFrame(0,columns=returns.columns, index=returns.index[23:])
    for i in range(23,len(returns)):
        df_clust.iloc[i-23,:]=clusterisation(returns.iloc[:i,:])
    return df_clust


df_cluster=dataframe_clusters(returns)
print(df_cluster)



