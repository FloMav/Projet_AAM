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
#plt.savefig('Data/books_read.jpeg', bbox_inches='tight', dpi=300)
#plt.show()

clusters = fcluster(output, t=1.3, criterion='distance')
df_clusters = pd.DataFrame(index=label)
df_clusters['clusters'] = clusters

df_clusters_1 = df[df_clusters[df_clusters['clusters'] == 1].index.values]
df_clusters_2 = df[df_clusters[df_clusters['clusters'] == 2].index.values]

def twelve_month_momentum(df):
    df_output = df.rolling(2).apply(lambda x: x.mean())
    return df_output

print(df_clusters_1.head(10))
print(twelve_month_momentum(df_clusters_1.head(10)))
# Momentum



