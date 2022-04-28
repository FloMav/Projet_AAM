import pandas as pd
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cut_tree
from matplotlib import pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from Partie_1 import clusterisation, dataframe_clusters

df = pd.read_csv("Data/DATA_PROJECT.csv", index_col=0)
corrMatrix = df.corr()

market = df.iloc[:, -1]
returns = df.drop(df.columns[-1], axis=1)

## Graph clusters at date t
features=preprocessing.scale(returns.iloc[:199,:],axis=1)
dist=pdist(features.T,'correlation')
output = linkage(dist,method='ward')
plt.figure(figsize=(10, 10))
csfont = {'fontname': 'Calibri', 'fontsize': '10'}
csfont2 = {'fontname': 'Calibri', 'fontsize': '8'}
csfont3 = {'fontname': 'Calibri', 'fontsize': '6'}
plt.xlabel('Distance', **csfont2)
plt.xticks(**csfont3)
plt.yticks(**csfont3)
dendrogram(output, color_threshold=1.5, truncate_mode='level', orientation='right', leaf_font_size=10, labels=returns.columns)
plt.savefig('Graphs/Cluster_t.png', bbox_inches='tight', dpi=300)


## Evolution of the cluster over time

df_cluster=dataframe_clusters(returns)

clust_1=[]
clust_2=[]

for i in range(len(df_cluster)):
    x1=0
    x2=0
    for col in df_cluster.columns:
        if df_cluster[col][i]==1:
            x1=x1+1
        else:
            x2=x2+1
    clust_1.append(x1)
    clust_2.append(x2)


from datetime import datetime

label=[]
for i in df_cluster.index:
    label.append(datetime.strptime(i, '%d/%m/%Y'))


file_path = 'Graphs/Clusters_evol.png'

plt.figure(figsize=(15, 10), dpi=80)
plt.plot(label,clust_1, color='red' , marker='o',label= 'Cluster 1')
plt.plot(label,clust_2, color='blue', marker='o',label='Cluster 2')
plt.legend()
plt.ylabel('Number of stocks per companies')
plt.title('Historical evolution of the number of stocks in the 2 clusters')
plt.savefig(file_path)