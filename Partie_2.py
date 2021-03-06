import numpy as np
import pandas as pd


def average_return(returns: pd.DataFrame) -> float:
    """
    :param returns: dataframe with the returns in the FIRST column.
    :return: the annualized average of the return.
    """
    returns = returns.values.tolist()  # Convert df to list
    average = float(np.mean(returns))*12
    return average


def volatility(returns: pd.DataFrame) -> float:
    """
    :param returns: dataframe with the returns in the FIRST column.
    :return: the annualized volatility of the return.
    """
    returns = returns.values.tolist()  # Convert df to list
    vol = (float(np.std(returns)))*np.sqrt(12)
    return vol


def VaR(returns, alpha: float = 0.95) -> float:
    """

    :param returns: dataframe with the returns in the FIRST column.
    :param alpha: confidence level of the VaR
    :return: the VaR at alpha % over month
    """
    returns = returns.iloc[:, 0]  # Convert dataframe to series
    var = returns.quantile(q=1-alpha)
    return var


def CVaR(returns, alpha: float = 0.95) -> float:
    """

    :param returns: dataframe with the returns in the FIRST column.
    :param alpha: confidence level of the CVaR
    :return: the CVaR at alpha % over month
    """
    var = VaR(returns, alpha=alpha)
    returns = returns.iloc[:, 0]  # Convert dataframe to series
    returns = returns[returns <= var]
    cvar = returns.mean()
    return cvar


def MDD(returns: pd.DataFrame, window=None) -> float:
    """
    :param returns: dataframe with the returns in the FIRST column.
    :param window: lookback window, int or None
     if None, look back entire history
    """
    track = returns.add(1).fillna(1).cumprod() * 100
    track = track.iloc[:, 0]  # Convert dataframe to series
    n = len(track)
    if window is None:
        window = n
    # rolling peak values
    peak_series = track.rolling(window=window, min_periods=1).max()
    return (track / peak_series - 1.0).min()


def luck(data, n_sim: int = 1000) -> np.ndarray:

    ret_sim = np.zeros((len(data), n_sim))
    stats_sim = np.zeros((5, n_sim))

    for i in range(n_sim):
        score_t = np.random.normal(0, 1, data.shape)
        for t in range(len(data)):
            long = score_t[t, :] >= np.quantile(score_t[t, :], 0.5)
            short = score_t[t, :] < np.quantile(score_t[t, :], 0.5)
            ret_sim[t, i] = np.mean(data[t, long]) - np.mean(data[t, short])

    luck_returns = pd.DataFrame(ret_sim)

    for i in range(n_sim):
        luck_returns_2 = pd.DataFrame(luck_returns.iloc[:, i])

        # annualized average returns
        stats_sim[0, i] = average_return(luck_returns_2)

        # annualized volatility
        stats_sim[1, i] = volatility(luck_returns_2)

        # VaR
        stats_sim[2, i] = VaR(luck_returns_2)

        # CVaR
        stats_sim[3, i] = CVaR(luck_returns_2)

        # MDD
        stats_sim[4, i] = MDD(luck_returns_2)

    #  STATISTICS LUCKY THRESHOLDS
    thresholds = np.quantile(stats_sim, 0.99, axis=1)

    return thresholds


df = pd.read_csv("Data/DATA_PROJECT.csv", index_col=0)
df_returns = np.array(df.drop(df.columns[-1], axis=1).iloc[36:, :])
print(luck(df_returns))

