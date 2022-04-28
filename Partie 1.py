import pandas as pd
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cut_tree
from matplotlib import pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

df = pd.read_csv("Data/DATA_PROJECT.csv", index_col=0)
corrMatrix = df.corr()

market = df.iloc[:, -1]
returns = df.drop(df.columns[-1], axis=1)
label = returns.columns

def clusterisation(returns: pd.DataFrame, n_clust=2):
    returns = preprocessing.scale(returns)
    dist = pdist(returns.T, 'correlation')
    output = linkage(dist, method='ward')
    clusters = fcluster(output, n_clust, criterion='maxclust')
    return clusters

def dataframe_clusters(returns: pd.DataFrame):
    df_clust = pd.DataFrame(0, columns=returns.columns, index=returns.index[36:])
    for i in range(36, len(returns)):
        df_clust.iloc[i - 36, :] = clusterisation(returns.iloc[:i, :])
    return df_clust

# plt.figure(figsize=(10, 10))
# csfont = {'fontname': 'Calibri', 'fontsize': '10'}
# csfont2 = {'fontname': 'Calibri', 'fontsize': '8'}
# csfont3 = {'fontname': 'Calibri', 'fontsize': '6'}
# plt.xlabel('Distance', **csfont2)
# plt.xticks(**csfont3)
# plt.yticks(**csfont3)
# dendrogram(output, color_threshold=1.5, truncate_mode='level', orientation='right', leaf_font_size=10, labels=label)
# plt.savefig('Data/books_read.jpeg', bbox_inches='tight', dpi=300)
# plt.show()

df = dataframe_clusters(returns)
#print(df)

def z_score():


def twelve_month_momentum_score(df: pd.DataFrame, df_clusters: pd.DataFrame):
    df_output = df.copy()
    #  12-month return momentum
    df_inter = df.shift().rolling(11).apply(lambda x: x.mean())
    #  within-cluster cross-sectional z-score
    for r in range(df.shape[0]):
        filter_1 = (df_clusters == 2).any()
        columns_drop_one = df_clusters.loc[df_clusters.index.values[r], filter_1].columns.values
        df_inter_1 = df_inter.iloc[r].drop(columns_drop_one, axis=1)
        df_inter_1 = pd.DataFrame(stats.zscore(df_inter_1, axis=None), columns=df_inter_1.columns,
                                  index=df_inter_1.index)

        filter_2 = (df_clusters == 1).any()
        columns_drop_two = df_clusters.loc[df_clusters.index.values[r], filter_2].columns.values
        df_inter_2 = df_inter.iloc[r].drop(columns_drop_two, axis=1)
        df_inter_2 = pd.DataFrame(stats.zscore(df_inter_2, axis=None), columns=df_inter_2.columns,
                                  index=df_inter_2.index)

        df_output.iloc[r] = pd.merge(df_inter_1, df_inter_2)

    return df_output


df_inter = twelve_month_momentum_score(returns, df)
print(df_inter)

# print(df_clusters_1.head(20))
# df_test = twelve_month_momentum_score(df_clusters_1.head(20))
# df_test.to_csv('Data/test.csv')
# print(twelve_month_momentum(df_clusters_1.head(20)))

def twelve_month_specific_momentum_score(df: pd.DataFrame):
    #  12-month specific momentum
    alpha_beta_error = {}
    df_output = pd.DataFrame(index=df.index, columns=df.columns)
    for c in df.columns:
        alpha_beta_error[c] = pd.DataFrame(
            RollingOLS(df[c].values, sm.add_constant(market.values), window=36).fit().params)
        alpha_beta_error[c][2] = df[c] - (alpha_beta_error[c][0] + alpha_beta_error[c][1] * market)
        df_output[c] = alpha_beta_error[c][0] + alpha_beta_error[c][2].rolling(12).mean()
    #  within-cluster cross-sectional z-score
    # df_output = stats.zscore(df_output, axis=1)
    return df_output[c]

# df_test = twelve_month_specific_momentum_score(df_clusters_1)
# print(df_test)

# df_test.to_csv('Data/test2.csv')


