import numpy as np
import pandas as pd


def average_return(track: pd.DataFrame) -> float:
    """
    :param track: dataframe with the index TRACK in the FIRST column.
    :return: the average return per period over the whole track.
    """
    track = track.iloc[:, 0].values.tolist()  # Convert df to list
    ave_ret = np.exp(np.log(track[-1]) / (len(track)-1)) - 1
    return ave_ret


def volatility(track: pd.DataFrame, nb_day: int = 252) -> float:
    """
    :param track: dataframe with the index TRACK in the FIRST column.
    :param nb_day: period of volatility computation.
    :return: the volatility of the return from the (end-nb_day) to the end of the track.
    """
    track = track.pct_change().dropna(axis=0)  # Convert track to returns
    track = track.iloc[-nb_day:, 0].values.tolist()  # Convert df to list taking into account nb_day
    vol = float(np.std(track))
    return vol


def VaR(track, alpha: float = 0.95, month: int = 21) -> float:
    """

    :param track: dataframe with the index TRACK in the FIRST column.
    :param alpha: confidence level of the VaR
    :param month: VaR over 1 month equivalent to 21 days (252/12)
    :return: the VaR at alpha % over month
    """
    track = track.pct_change().dropna(axis=0)  # Convert track to returns
    track = track.iloc[-month:, 0]  # Convert dataframe to series
    var = track.quantile(q=1-alpha, interpolation='lower')
    return var


def CVaR(track, alpha: float = 0.95, month: int = 21) -> float:
    """

    :param track: dataframe with the index TRACK in the FIRST column.
    :param alpha: confidence level of the CVaR
    :param month: CVaR over 1 month equivalent to 21 days (252/12)
    :return: the CVaR at alpha % over month
    """
    var = VaR(track, alpha=alpha, month=month)
    track = track.pct_change().dropna(axis=0)  # Convert track to returns
    track = track.iloc[:, 0]  # Convert dataframe to series
    track = track[track <= var]
    cvar = track.mean()
    return cvar


def MDD(track: pd.DataFrame, window=None) -> float:
    """
    :param track: dataframe with the index TRACK in the FIRST column.
    :param window: lookback window, int or None
     if None, look back entire history
    """
    track = track.iloc[:, 0]  # Convert dataframe to series
    n = len(track)
    if window is None:
        window = n
    # rolling peak values
    peak_series = track.rolling(window=window, min_periods=1).max()
    print(peak_series)
    return (track / peak_series - 1.0).min()


# df = pd.DataFrame([1, 1.2, 1.5, 1.4, 1.39, 1.35, 1.3, 1.23, 1.22, 1.11, 1.08, 1.34, 1.05, 0.9, 1.6], columns=['track'])
# print(df)
# print(MDD(df))




