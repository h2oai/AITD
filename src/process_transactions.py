import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta

# local imports
from src import config as dc
from src.sentiment import *
from src.process_transactions import *


def pre_process_txn(df: pd.DataFrame) -> pd.DataFrame:
    """Reads transactions from a csv file into a pandas df
    Parameters:
        df (pd.DataFrame): input dataframe
    Returns:
        pd.DataFrame: Pandas DF containing all the transactions
    """
    dc.logger.info("Preprocessing data...")
    df[dc.txn_data_ind_col] = df.index
    df[dc.txn_data_amount_col] = (
        df[dc.txn_data_amount_col]
        .replace(",", "", regex=True)
        .apply(pd.to_numeric, errors="coerce")
    )
    df[["tx_date"]] = df[["tx_date"]].apply(pd.to_datetime)
    dc.logger.info(f"Converting amount to float; df is now has shape {str(df.shape)}")

    return df

def sanity_check(df: pd.DataFrame, score_month: int = None, score_year: int = None, lag: int = 2) -> None:
    """Warns user if some data will be discard
    Parameters:
        df (pd.DataFrame): input dataframe
    Returns:
        pd.DataFrame: Pandas DF containing all the transactions
    """
    
    dc.logger.info("Data sanity check...")
    if score_month is None: 
        date_r = df["tx_date"].max()
        dc.logger.info("Score month not Specified, using month " + str(date_r.month))
    else: 
        if score_year is None: 
            date_r = max(df[df['tx_date'].dt.month == score_month]['tx_date'])
        else: 
            date_r = max(df[(df['tx_date'].dt.month == score_month) & (df['tx_date'].dt.year == score_year)]['tx_date'])
    
    date_l = (date_r - relativedelta(months=(lag))).replace(day=1)
    if_discard = len(df) - len(df[(df["tx_date"] > date_l) & (df["tx_date"] <= date_r)])

    # Inform User about options selected
    if if_discard:
        msg = f"Found latest transaction date: {date_r:%Y-%m-%d}, dicard {if_discard} data points with date before {date_l:%Y-%m-%d}. If a sender-receiver relation only exist in data prior to {date_l:%Y-%m-%d} or after {date_r:%Y-%m-%d}, they won't appear in the final output"
        dc.logger.warn(msg)
        print(f"Warning: {msg}")
    


def merge_tox_amo_scores(
    tox_df: pd.DataFrame,
    emo_df: pd.DataFrame,
    text_feat_df: pd.DataFrame,
    sent_df: pd.DataFrame,
) -> pd.DataFrame:
    merged_tox_emo_df = pd.DataFrame()
    # print(emo_df)
    # merging emotions / toxicity / both results
    merged_tox_emo_df = pd.merge(tox_df, emo_df, on=dc.txn_data_ind_col)
    sent_df.drop(
        columns=["sender_id", "receiver_id", "tx_description", "amount", "tx_date"],
        inplace=True,
    )
    merged_text_sent_df = pd.merge(text_feat_df, sent_df, on=dc.txn_data_ind_col)
    prefinal_set = pd.merge(
        merged_tox_emo_df, merged_text_sent_df, on=dc.txn_data_ind_col
    )

    prefinal_set.fillna(0, inplace=True)
    prefinal_set.drop_duplicates(subset=dc.txn_data_ind_col, inplace=True)
    return prefinal_set


def get_prev_year_months(year: int, month: int, n: int = 2) -> list:
    res = []
    year = f"{year}"
    month = f"{month}".rjust(2, "0")
    prev_month = datetime.datetime.strptime(f"{year}-{month}-01", "%Y-%m-%d")
    while n > 0:
        prev_month = prev_month - datetime.timedelta(days=1)
        prev_month = prev_month.replace(day=1)
        res.append(prev_month.year * 100 + prev_month.month)
        n -= 1
    return res


def construct_feature_df(
    final_edges: pd.DataFrame,
    year: int,
    month: int,
    lag: int = 2,
    reciprical_features: bool = True,
    lag_features=True,
) -> pd.DataFrame:
    
    month_ini = pd.DataFrame()
    reference = dc.reference_features
    cols_to_duplicate = set(reference).intersection(final_edges.columns)
    year_month = int(year) * 100 + int(month)
    month_ini = final_edges[final_edges["year_month"] == year_month].copy()

    assert (
        month_ini.shape[0] > 0
    ), f"no data available with year_month filter {year_month}"

    if reciprical_features:
        month_inid = month_ini[cols_to_duplicate].copy()
        month_inid.columns = [
            ("recip_" + str(i)) if i not in (["sender_id", "receiver_id"]) else i
            for i in list(month_inid.columns)
        ]
        month_ini.drop_duplicates(inplace=True)
        month_ini = pd.merge(
            month_ini,
            month_inid,
            left_on=["sender_id", "receiver_id"],
            right_on=["sender_id", "receiver_id"],
            how="left",
        )
        month_ini.drop_duplicates(inplace=True)
        """
        month_ini.drop(['sender_id_y', 'usr_recip_y'], axis=1, inplace=True)
        month_ini.rename(columns={
            'sender_id_x': 'sender_id',
            'receiver_id_x': 'receiver_id'
        },  inplace=True)
        """
    if lag > 0 and lag_features:
        year_months = get_prev_year_months(year, month, n=lag)
        # print(year_months)
        for month_offset, year_month in enumerate(year_months):
            month_offset += 1
            month_n = final_edges[final_edges["year_month"] == year_month][
                cols_to_duplicate
            ].copy()
            assert (
                month_n.shape[0] > 0
            ), f"no data available with year_mont filter {year_month}"
            month_n.columns = [
                f"month_min{month_offset}_{str(i)}"
                if i not in (["sender_id", "receiver_id"])
                else i
                for i in list(month_n.columns)
            ]
            month_ini = pd.merge(
                month_ini, month_n, on=["sender_id", "receiver_id"], how="left"
            )
        month_ini.fillna(0, inplace=True)
        return month_ini
    return month_ini


def gen_features(final_edges: pd.DataFrame, txn_df: pd.DataFrame, lag: int, score_month: int, score_year: int) -> pd.DataFrame:
    final_normalised_edges = normalize_features(final_edges)

    feature_df = pd.DataFrame()

    if score_year is None:
        year = max(txn_df["tx_date"]).year
    else: 
        year = score_year
    
    # print(year)
    if score_month is None:
        month = max(txn_df["tx_date"]).month
    else: 
        month = score_month
    # print(month)
    lag = lag

    feature_df = construct_feature_df(
        final_normalised_edges, year, month, lag, True, True
    )
    return feature_df
