import pandas as pd
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

df = pd.read_csv("Subject/DATA_PROJECT.csv", index_col=0, sep=';')

for col in df.columns[0:]:
    df[col]=df[col].apply(lambda x:float(x))

corrMatrix = df.corr()
print(corrMatrix)

returns=preprocessing.scale(df)

dist=pdist(returns.T,'correlation')
output = linkage(dist,method='ward')

# GRAPH
plt.figure(figsize=(10, 10))
csfont = {'fontname':'Calibri', 'fontsize' : '10'}
csfont2 = {'fontname':'Calibri', 'fontsize' : '8'}
csfont3 = {'fontname':'Calibri', 'fontsize' : '6'}
plt.xlabel('Distance',**csfont2)
plt.xticks(**csfont3)
plt.yticks(**csfont3)
dendrogram(output,color_threshold=1.5,truncate_mode='level',orientation='right',leaf_font_size=10,labels=df.columns[0:])
plt.savefig('books_read.jpeg',bbox_inches='tight', dpi=300)
plt.show()

