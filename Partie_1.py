import pandas as pd
from sklearn import preprocessing
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from statsmodels.regression.rolling import RollingOLS

# Import dataframe
df = pd.read_csv("Data/DATA_PROJECT.csv", index_col=0)
corrMatrix = df.corr()

df_market = df.iloc[:, -1]
df_returns = df.drop(df.columns[-1], axis=1)


def clusterisation(returns_range: pd.DataFrame) -> np.ndarray:
    """
    :param returns_range: a data range used to create the clusters at date t
    :return: an array of 1 and 2 with shape  (1,len(returns_range.columns))
    """
    returns_range = preprocessing.scale(returns_range, axis=1)
    dist = pdist(returns_range.T, 'correlation')
    output = linkage(dist, method='ward')
    n_clust = 2
    clusters = fcluster(output, n_clust, criterion='maxclust')
    return clusters


def dataframe_clusters(df_ret: pd.DataFrame) -> pd.DataFrame:
    """
    For the first date t=35, the function takes 35 lines of data to create the clusters. For the second the date t=36,
    the function takes 36 lines of datas and so on.
    :param df_ret: the initial dataframe with the returns for all the tickers without the market
    :return: an
    """
    df_clust = pd.DataFrame(0, columns=df_ret.columns, index=df_ret.index[35:])
    for i in range(35, len(df_ret)):
        df_clust.iloc[i - 35, :] = clusterisation(df_ret.iloc[:i, :])
    return df_clust


def z_score(df_value: pd.DataFrame, df_clust: pd.DataFrame) -> pd.DataFrame:
    """
    The function loops on each rows of the dataframe and create two different series according to the clusters. Then it
    applies the z-score on each series separately and recombine them.
    :param df_value: dataframe containing the value on which to apply the z-score
    :param df_clust: same dataframe above but with the clusters instead of the values
    :return: same dataframe as above but with the within-cluster cross-sectional z-score applied
    """
    #print(df_value)
    #print(df_clust)
    df_value = df_value.apply(pd.to_numeric, errors='coerce')
    df_output = pd.DataFrame(columns=df_value.columns)
    for r in range(df_value.shape[0]):
        df_output_inter = df_value.iloc[[r]]
        df_inter = df_clust.iloc[[r]]

        columns_1 = df_inter[df_inter == 1].dropna(axis=1).columns.values
        columns_2 = df_inter[df_inter == 2].dropna(axis=1).columns.values

        df_output_1 = df_output_inter.drop(columns_2, axis=1)
        df_output_2 = df_output_inter.drop(columns_1, axis=1)

        df_output_1 = stats.zscore(df_output_1, axis=None)
        df_output_2 = stats.zscore(df_output_2, axis=None)

        df_output_inter = pd.concat([df_output_1, df_output_2], axis=1)
        df_output = pd.concat([df_output, df_output_inter], axis=0)

    return df_output


print()


def R_MOM(df_ret: pd.DataFrame) -> pd.DataFrame:
    """
    - Do the r_mom(s,t)
    - Apply the within-cluster cross sectional z-score
    :param df_ret: the initial dataframe with the returns for all the tickers without the market
    :return: a dataframe of the same size as the initial (index, columns) with computation done for the R_MOM(s,t)
    """
    # r_mom 12-month return momentum
    df_output = df_ret.shift().rolling(11).apply(lambda x: x.mean())
    df_output = df_output.iloc[35:, :]  # dataframe that starts at 31/07/2008

    #  within-cluster cross-sectional z-score
    df_cluster = dataframe_clusters(df_ret)  # dataframe that starts at 31/07/2008
    df_output = z_score(df_output, df_cluster)
    return df_output


def s_mom(df_ret: pd.DataFrame) -> pd.DataFrame:
    """
    For each columns c:
        - We perform a rolling regression on a 36 windows. We get as many alphas and betas as there are 36 windows in
        the column.
        - Then, we retrieve the errors. For each alpha and beta, we retrieve the last 12 errors (errors computed with
        those specific alpha and beta).
    :param df_ret: the initial dataframe with the returns for all the tickers without the market
    :return: a dataframe of the same size as the initial (index, columns) with computation done for the s_mom(s,t)
    """
    alpha_beta_error = {}
    error = {}
    df_output = pd.DataFrame(index = df.index, columns=df.columns)
    for c in df.columns:
        #Retrive the alpha and beta for a 36 windows for the whole list
        alpha_beta_error[c] = pd.DataFrame(RollingOLS(df[c].values, sm.add_constant(df_market.values), window=36).fit().params)

        #Compute the errors for each alpha and beta
        error[c] = {}
        for model in range(len(alpha_beta_error[c].index)):
            error[c][model] = {}
            for i in range(12):
                error[c][model][i] = df[c][model-i] - (alpha_beta_error[c][0][model] + alpha_beta_error[c][1][model] * df_market[model-i])
            df_output[c][model] = alpha_beta_error[c][0][model] + np.array(list(error[c][model].values())).mean()
    return df_output


def S_MOM(df_ret: pd.DataFrame) -> pd.DataFrame:
    """
    - Do the s_mom(s,t)
    - Apply the within-cluster cross sectional z-score
    :param df_ret: the initial dataframe with the returns for all the tickers without the market
    :return: a dataframe of the same size as the initial (index, columns) with computation done for the S_MOM(s,t)
    """
    # s_mom 12-month return momentum
    df_output = s_mom(df_ret)
    df_output = df_output.iloc[35:, :]  # dataframe that starts at 31/07/2008
    df_output.drop('EUROSTOXX 50 TR index', axis=1, inplace=True)  # dataframe that starts at 31/07/2008
    #  within-cluster cross-sectional z-score
    df_cluster = dataframe_clusters(df_ret)  # dataframe that starts at 31/07/2008
    df_output = z_score(df_output, df_cluster)
    return df_output


#print(S_MOM(df_returns))












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

    w_global = row_weights.copy()
    w_global.iloc[col_1] = row_weights.iloc[col_1].values*N1/(N1+N2)
    w_global.iloc[col_2] = row_weights.iloc[col_2].values*N2/(N1+N2)

    return w_global

def dataframe_global_weights(df_cluster, df_weights):
    df_global_weights=pd.DataFrame(0,columns=returns.columns, index=returns.index[36:])
    for i in range(36,len(returns)):
        df_global_weights.iloc[i - 36, :]=global_port(df_cluster.iloc[i-36,:],df_weights.iloc[i-36,:])

    return df_global_weights

df_glob_weights=dataframe_global_weights(df_cluster,df_weights)
print(df_glob_weights)

# check si les poids somment bien à 1 au niveau global
# for i in range(len(df_glob_weights)):
#     print(df_glob_weights.iloc[i,:].sum())

#step 5

