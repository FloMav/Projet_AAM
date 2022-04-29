import pandas as pd
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cut_tree
from matplotlib import pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.regression.linear_model import OLS


df = pd.read_csv("Data/DATA_PROJECT.csv", index_col=0)
corrMatrix = df.corr()

market = df.iloc[:, -1]
returns = df.drop(df.columns[-1], axis=1)

## construction data frame avec les clusters
def clusterisation(returns):
    returns=preprocessing.scale(returns, axis=1)
    dist = pdist(returns.T, 'correlation')
    output = linkage(dist, method='ward')

    n_clust=2
    clusters = fcluster(output, n_clust,criterion='maxclust')

    return  clusters

def dataframe_clusters(returns):
    df_clust=pd.DataFrame(0,columns=returns.columns, index=returns.index[36:])
    for i in range(36,len(returns)):
        df_clust.iloc[i-36,:]=clusterisation(returns.iloc[:i,:])
    return df_clust

# identification des indices de colonnes pour chaque cluster
def fonction_moise(aray_1_2):
    col_1 = []
    col_2 = []
    for i in range(len(aray_1_2)):
        if aray_1_2[i] == 1:
            col_1.append(i)
        else:
            col_2.append(i)

    return col_1, col_2

def recombinator(array_1,array_2,col_1,col_2):
    combinator=np.zeros(46)
    v1 = 0
    for i in col_1:
        combinator[i] = array_1[v1]
        v1 = 1 + v1

    v2 = 0
    for i in col_2:
        combinator[i] = array_2[v2]
        v2 = 1 + v2

    return combinator

## calcul du z_score pour r mom
def r_mom(df_returns, i):
    return df_returns.iloc[i - 11:i].mean()

def z_score(array_1_2, array_values):
    col_1, col_2 = fonction_moise(array_1_2)

    val_1 = preprocessing.scale(array_values.iloc[col_1], axis=0)
    val_2 = preprocessing.scale(array_values.iloc[col_2], axis=0)

    z_scores = recombinator(val_1,val_2,col_1,col_2)

    return z_scores

def dataframe_Rmom(returns, df_cluster):
    df_Rmom=pd.DataFrame(0,columns=returns.columns, index=returns.index[36:])
    for i in range(36,len(returns)):
        rmom=r_mom(returns.iloc[:i,:],i)
        df_Rmom.iloc[i - 36, :]=z_score(df_cluster.iloc[i-36,:],rmom)
    return df_Rmom

df_cluster=dataframe_clusters(returns)
print(df_cluster)

df_Rmom= dataframe_Rmom(returns,df_cluster)
print(df_Rmom)

#df_Rmom.to_csv('Data\Rmom_test.csv', sep=';')

## calcul du z_score pour s mom
def s_mom(df_returns, market, i):
    s_moms = []
    for col in range(len(df_returns.columns)):
        model = OLS(df_returns.iloc[i - 36:i, col].values, sm.add_constant(market.iloc[i - 36:i].values)).fit()
        alpha = model.params[0]
        sum_resid = model.resid[-12:].sum()
        s_mom = (12 * alpha + sum_resid) / 12
        s_moms.append(s_mom)

    return pd.Series(s_moms)

def dataframe_Smom(returns, df_cluster):
    df_Smom=pd.DataFrame(0,columns=returns.columns, index=returns.index[36:])
    for i in range(36,len(returns)):
        smom=s_mom(returns.iloc[:i,:], market,i)
        df_Smom.iloc[i - 36, :]=z_score(df_cluster.iloc[i-36,:],smom)
    return df_Smom

df_Smom= dataframe_Smom(returns,df_cluster)
print(df_Smom)

def dataframe_Mom(df_Smom,df_Rmom):
    df_Mom = pd.DataFrame(0, columns=df_Smom.columns, index=df_Smom.index)
    df_Mom=(df_Smom+df_Rmom)/2
    return df_Mom

df_Mom=dataframe_Mom(df_Smom,df_Rmom)
print(df_Mom)

# étape 3

def long_short(row_cluster_score, returns_cluster):
    median = row_cluster_score.median()
    weights = pd.Series(0, index=returns_cluster.columns)

    col_long = row_cluster_score[row_cluster_score >= median].index
    col_short = row_cluster_score[row_cluster_score < median].index

    cov_long = np.cov(returns_cluster.drop(col_short, axis=1), rowvar=False)
    cov_short = np.cov(returns_cluster.drop(col_long, axis=1), rowvar=False)

    InvVolWeightAssets_long = 1 / np.sqrt(np.diagonal(cov_long))
    InvVolWeightAssets_short = 1 / np.sqrt(np.diagonal(cov_short))

    SumInvVolWeightAssets_long = np.sum(1 / np.sqrt(np.diagonal(cov_long)))
    SumInvVolWeightAssets_short = np.sum(1 / np.sqrt(np.diagonal(cov_short)))

    for i in range(len(col_long)):
        weights.loc[col_long[i]] = InvVolWeightAssets_long[i] / SumInvVolWeightAssets_long

    for i in range(len(col_short)):
        weights.loc[col_short[i]] = -InvVolWeightAssets_short[i] / SumInvVolWeightAssets_short

    return weights

def dataframe_weights(returns,df_cluster, df_Mom):
    df_Weights=pd.DataFrame(0,columns=returns.columns, index=returns.index[36:])
    for i in range(36,len(returns)):
        col_1, col_2 = fonction_moise(df_cluster.iloc[i-36,:])

        weights_1 = long_short(df_Mom.iloc[i-36, col_1], returns.iloc[:i,col_1])
        weights_2 = long_short(df_Mom.iloc[i-36, col_2], returns.iloc[:i,col_2])

        weights=recombinator(weights_1,weights_2,col_1,col_2)

        df_Weights.iloc[i - 36, :]=weights
    return df_Weights

df_weights=dataframe_weights(returns,df_cluster,df_Mom)
print(df_weights)

# check si les poids somment bien à 1 au niveau global
# for i in range(len(df_weights)):
#     print(df_weights.iloc[i,:].sum())

#étape 4

def global_port(row_cluster, row_weights):
    col_1,col_2 = fonction_moise(row_cluster)

    N1=len(col_1)
    N2=len(col_2)

    row_weights.iloc[col_1] = row_weights.iloc[col_1]*N1/(N1+N2)
    row_weights.iloc[col_2] = row_weights.iloc[col_2]*N2/(N1+N2)

    return row_weights

print(global_port(df_cluster.iloc[0,:],df_weights.iloc[0,:]))