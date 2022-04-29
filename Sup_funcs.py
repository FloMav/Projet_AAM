import pandas as pd


def pivot_table(df_in: pd.DataFrame, df_out: pd.DataFrame, threshold: tuple, equal=1):
    if equal:
        columns_1 = df_in[df_in == threshold[0]].dropna(axis=1).columns.values
        columns_2 = df_in[df_in == threshold[1]].dropna(axis=1).columns.values
        df_1 = df_out[columns_1]
        df_2 = df_out[columns_2]
    else:
        columns_1 = df_in[df_in >= threshold[0]].dropna(axis=1).columns.values
        columns_2 = df_in[df_in < threshold[0]].dropna(axis=1).columns.values
        df_1 = df_out[columns_1].transpose()
        df_2 = df_out[columns_2].transpose()

    return df_1, df_2
