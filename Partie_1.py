import pandas as pd
from sklearn import preprocessing
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from statsmodels.regression.rolling import RollingOLS
from matplotlib import pyplot as plt
import Sup_funcs as sf

# Import dataframe
df = pd.read_csv("Data/DATA_PROJECT.csv", index_col=0)
corrMatrix = df.corr()

df_market = df.iloc[:, -1]
df_returns = df.drop(df.columns[-1], axis=1)


################################################################# STEP 1 ###############################################

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


def dataframe_clusters() -> pd.DataFrame:
    """
    For the first date t=35, the function takes 36 lines of data to create the clusters. For the second the date t=36,
    the function takes 37 lines of datas and so on.
    :return: a dataframe with the clusters for each tickers at each date.
        The index starts at 31/07/2008 and there is all the columns except the market.
    """

    df_clust = pd.DataFrame(0, columns=df_returns.columns, index=df_returns.index[35:])
    for i in range(35, len(df_returns)):
        df_clust.iloc[i - 35, :] = clusterisation(df_returns.iloc[:i, :])

    return df_clust


df_clusters = dataframe_clusters()


################################################################# STEP 2 ###############################################

def z_score(df_value: pd.DataFrame) -> pd.DataFrame:
    """
    The function loops on each rows of the dataframe and create two different series according to the clusters. Then it
    applies the z-score on each series separately and recombine them.
    :param df_value: dataframe containing the value on which to apply the z-score.
    :return: same dataframe as above but with the within-cluster cross-sectional z-score applied.
    """

    # print(df_value)
    df_value = df_value.apply(pd.to_numeric, errors='coerce')
    df_output = pd.DataFrame(columns=df_value.columns)
    for r in range(df_value.shape[0]):
        df_output_inter = df_value.iloc[[r]]
        df_inter = df_clusters.iloc[[r]]

        df_output_1, df_output_2 = sf.pivot_table(df_inter, df_output_inter, (1, 2))

        df_output_1 = stats.zscore(df_output_1, axis=None)
        df_output_2 = stats.zscore(df_output_2, axis=None)

        df_output_inter = pd.concat([df_output_1, df_output_2], axis=1)
        df_output = pd.concat([df_output, df_output_inter], axis=0)

    return df_output


#print(z_score(df_returns.iloc[35:, :]))


def R_MOM() -> pd.DataFrame:
    """
    - Do the 𝑟_𝑚𝑜𝑚𝑠,𝑡.
    - Apply the within-cluster cross sectional z-score.
    :return: a dataframe with computation done for the (𝑅_𝑀𝑂𝑀𝑠,𝑡).
        The index starts at 31/07/2008 and there is all the columns except the market.
    """

    # 𝑟_𝑚𝑜𝑚𝑠,𝑡 12-month return momentum
    df_output = df_returns.shift().rolling(11).apply(lambda x: x.mean())
    df_output = df_output.iloc[35:, :]  # dataframe that starts at 31/07/2008

    #  within-cluster cross-sectional z-score
    df_output = z_score(df_output)

    return df_output


# print(R_MOM())


def s_mom() -> pd.DataFrame:
    """
    For each columns c:
        - We perform a rolling regression on a 36 windows. We get as many alphas and betas as there are 36 windows in
        the column.
        - Then, we retrieve the errors. For each alpha and beta, we retrieve the last 12 errors (errors computed with
        those specific alpha and beta).
    :return:  a dataframe with computation done for the 𝑠_𝑚𝑜𝑚𝑠,𝑡.
        The index starts at 31/07/2008 and there is all the columns except the market.
    """

    alpha_beta_error = {}
    error = {}
    df_output = pd.DataFrame(index=df_returns.index, columns=df_returns.columns)

    for c in df_output.columns:
        # Retrieve the alpha and beta for a 36 windows for the whole list
        alpha_beta_error[c] = pd.DataFrame(RollingOLS(df_returns[c].values, sm.add_constant(df_market.values),
                                                      window=36).fit().params)

        # Compute the errors for each alpha and beta
        error[c] = {}
        for model in range(len(alpha_beta_error[c].index)):  # Loop in all the different alpha/beta for each date.
            error[c][model] = {}
            for i in range(12):  # Loop in the last 11 + current errors for a couple alpha/beta
                error[c][model][i] = df_returns[c][model - i] - (alpha_beta_error[c][0][model] +
                                                                 alpha_beta_error[c][1][model] * df_market[model - i])
            df_output[c][model] = alpha_beta_error[c][0][model] + np.array(list(error[c][model].values())).mean()

    return df_output


# print(s_mom())


def S_MOM() -> pd.DataFrame:
    """
    - Do the 𝑠_𝑚𝑜𝑚𝑠,𝑡.
    - Apply the within-cluster cross sectional z-score.
    :return: a dataframe with computation done for the (𝑆_𝑀𝑂𝑀𝑠,𝑡).
            The index starts at 31/07/2008 and there is all the columns except the market.
    """

    # 𝑠_𝑚𝑜𝑚𝑠,𝑡 12-month return momentum
    df_output = s_mom()
    df_output = df_output.iloc[35:, :]  # dataframe that starts at 31/07/2008

    #  within-cluster cross-sectional z-score
    df_output = z_score(df_output)

    return df_output


# print(S_MOM())


def MOM(SMOM: pd.DataFrame, RMOM: pd.DataFrame) -> pd.DataFrame:
    """
    𝑀𝑂𝑀𝑠,𝑡 = (𝑅_𝑀𝑂𝑀𝑠,𝑡 + 𝑆_𝑀𝑂𝑀𝑠,𝑡)/2
    :param SMOM: dataframe with the 12-months specific momentum score (𝑆_𝑀𝑂𝑀𝑠,𝑡).
    :param RMOM: dataframe with the 12-months return momentum score (𝑅_𝑀𝑂𝑀𝑠,𝑡).
    :return: a dataframe with computation done for the (𝑀𝑂𝑀𝑠,𝑡).
            The index starts at 31/07/2008 and there is all the columns except the market.
    """

    MOM = (SMOM + RMOM) / 2

    return MOM


df_R_MOM = R_MOM()
df_S_MOM = S_MOM()
df_MOM = MOM(df_R_MOM, df_S_MOM)


################################################################# STEP 3 ###############################################

def long_short_weights() -> pd.DataFrame:
    """
    For each row r:
        - Retrieve the clusters on the dataframe containing the MOM scores. We get the two clusters with the score of
            each stock in it.
        - Within each cluster, we split the stock and their MOM score according to the median of the cluster
            into 2 sub category Long/Short.
        - Switch to the expanding volatility dataframe, we apply the inverse volatility weighting scheme within
            each sub-category.
            --> Cluster 1 / Long : total weight = 1
            --> Cluster 1 / Short : total weight = -1
            --> Cluster 1 : total weight = 0
            --> Cluster 2 / Long : total weight = 1
            --> Cluster 2 / Short : total weight = -1
            --> Cluster 2 : total weight = 0
            --> Portfolio : total weight = 0
    :return: a dataframe from 31/07/2008 with the weight of each stock at each date.
    """

    df_w = pd.DataFrame(columns=df_returns.columns, index=df_returns.index[35:])
    #df_vol = df_returns.expanding().std().iloc[35:, :]
    df_vol = df_returns.rolling(5).std().iloc[35:, :]

    for r in range(df_w.shape[0]):
        df_MOM_inter = df_MOM.iloc[[r]]
        df_clust_inter = df_clusters.iloc[[r]]
        df_vol_inter = df_vol.iloc[[r]]

        df_MOM_1, df_MOM_2 = sf.pivot_table(df_clust_inter, df_MOM_inter, (1, 2))

        ####################### CREATING THE LONG/SHORT PER CLUSTER #######################
        df_vol_1_long, df_vol_1_short = sf.pivot_table(df_MOM_1, df_vol_inter, (np.median(df_MOM_1.values),),
                                                       equal=0)
        df_vol_2_long, df_vol_2_short = sf.pivot_table(df_MOM_2, df_vol_inter, (np.median(df_MOM_2.values),),
                                                       equal=0)
        ###################################################################################

        ############# COMPUTE THE INV VOL WEIGHT PER LONG/SHORT PER CLUSTER ###############
        def inv_vol_weight(x): return (1 / x) / np.sum((1 / x))

        df_inv_vol_weight_1_long = df_vol_1_long.apply(inv_vol_weight).transpose()
        #print(df_inv_vol_weight_1_long)
        df_inv_vol_weight_1_short = df_vol_1_short.apply(inv_vol_weight).transpose() * -1
        #print(df_inv_vol_weight_1_short.sum())
        df_inv_vol_weight_2_long = df_vol_2_long.apply(inv_vol_weight).transpose()
        #print(df_inv_vol_weight_2_long.sum())
        df_inv_vol_weight_2_short = df_vol_2_short.apply(inv_vol_weight).transpose() * -1
        #print(df_inv_vol_weight_2_short.sum())
        ###################################################################################

        df_weights_inter = pd.concat([df_inv_vol_weight_1_long, df_inv_vol_weight_1_short,
                                      df_inv_vol_weight_2_long, df_inv_vol_weight_2_short], axis=1)

        df_w = pd.concat([df_w, df_weights_inter], axis=0)
        df_w.dropna(axis=0, inplace=True)  # Index mismatch because of transpose

    return df_w


df_weights = long_short_weights()

# check si les poids somment bien à 1 au niveau global
# for i in range(len(df_weights)):
#      print(df_weights.iloc[i,:].sum())

################################################################# STEP 4 ###############################################
def global_weights() -> pd.DataFrame:
    """
    For each row r:
        - Retrieve the 2 dataframes of normal weights for the clusters
        - Apply the weighting scheme within each clusters.
            ???????
    :return: a dataframe from 31/07/2008 with the global weight of each stock at each date.
    """

    df_gw = pd.DataFrame(columns=df_returns.columns, index=df_returns.index[35:])

    for r in range(df_gw.shape[0]):

        df_clust_inter = df_clusters.iloc[[r]]
        df_w_inter = df_weights.iloc[[r]]

        df_w_1, df_w_2 = sf.pivot_table(df_clust_inter, df_w_inter, (1, 2))

        ############# COMPUTE THE INV VOL WEIGHT PER LONG/SHORT PER CLUSTER ###############
        def gw(x): return x * x.shape/46

        df_gw_1 = df_w_1.transpose().apply(gw).transpose()
        df_gw_2 = df_w_2.transpose().apply(gw).transpose()
        ###################################################################################

        df_gw_inter = pd.concat([df_gw_1, df_gw_2], axis=1)
        df_gw = pd.concat([df_gw, df_gw_inter], axis=0)
        df_gw.dropna(axis=0, inplace=True)  # Index mismatch because of transpose

    return df_gw


df_global_weights = global_weights()
#df_global_weights.to_csv('Data/gw.csv')
#print(df_global_weights)

# # check si les poids somment bien à 1 au niveau global
# for i in range(len(df_global_weights)):
#      print(df_global_weights.iloc[i, :].sum())

def track():
    df_ret = df_returns.iloc[36:, :]
    df_g_w = df_global_weights.shift().dropna(axis=0)
    df_track = df_g_w * df_ret
    df_track = df_track.sum(1).add(1).fillna(1).cumprod() * 100
    print(df_ret)
    print(df_g_w)
    print(df_track)
    market_port = df_market[36:].add(1).fillna(1).cumprod() * 100
    plt.plot(df_market.index[36:], df_track, color='red', label='strategy')
    plt.plot(df_market.index[36:], market_port, color='blue', label='market')
    plt.legend()
    plt.show()
    pass

track()


################################################################# STEP 5 ###############################################



