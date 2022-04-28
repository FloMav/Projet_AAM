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


def clusterisation(ret: pd.DataFrame, n_clust=2):
    ret = preprocessing.scale(ret)
    dist = pdist(ret.T, 'correlation')
    output = linkage(dist, method='ward')
    clusters = fcluster(output, n_clust, criterion='maxclust')
    return clusters


def dataframe_clusters(ret: pd.DataFrame):
    df_clust = pd.DataFrame(0, columns=ret.columns, index=ret.index[36:])
    for i in range(36, len(ret)):
        df_clust.iloc[i - 36, :] = clusterisation(ret.iloc[:i, :])
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

df_clusters = dataframe_clusters(returns)


def z_score(df_value):
    df_output = df_value.copy()
    #  within-cluster cross-sectional z-score
    for r in range(df.shape[0]):
        filter_1 = (df_clusters == 2).any()
        columns_drop_one = df_clusters.loc[df_clusters.index.values[r], filter_1].index.values
        #df_inter_1 = pd.DataFrame(df_output.iloc[r].drop(columns_drop_one))
        #df_inter_1 = stats.zscore(df_inter_1.values, axis=None)
        print(columns_drop_one)

    return df_output


def twelve_month_momentum_score(df_mom_score: pd.DataFrame):
    #  12-month return momentum
    df_inter = df_mom_score.shift().rolling(11).apply(lambda x: x.mean())
    df_output = z_score(df_inter)
    return df_output


df_inter = twelve_month_momentum_score(returns)
print(df_inter)


# print(df_clusters_1.head(20))
# df_test = twelve_month_momentum_score(df_clusters_1.head(20))
# df_test.to_csv('Data/test.csv')
# print(twelve_month_momentum(df_clusters_1.head(20)))

def twelve_month_specific_momentum_score(df_spe_mom_score: pd.DataFrame):
    #  12-month specific momentum
    alpha_beta_error = {}
    df_output = pd.DataFrame(index=df_spe_mom_score.index, columns=df_spe_mom_score.columns)
    for c in df_spe_mom_score.columns:
        alpha_beta_error[c] = pd.DataFrame(
            RollingOLS(df_spe_mom_score[c].values, sm.add_constant(market.values), window=36).fit().params)
        alpha_beta_error[c][2] = df_spe_mom_score[c] - (alpha_beta_error[c][0] + alpha_beta_error[c][1] * market)
        df_output[c] = alpha_beta_error[c][0] + alpha_beta_error[c][2].rolling(12).mean()
    #  within-cluster cross-sectional z-score
    # df_output = stats.zscore(df_output, axis=1)
    return df_output[c]

# df_test = twelve_month_specific_momentum_score(df_clusters_1)
# print(df_test)

# df_test.to_csv('Data/test2.csv')
