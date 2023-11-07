import pandas as pd
import numpy as np
from typing import Tuple, Callable, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# local imports
from src import config as dc


def load_sentiment_model() -> Any:
    analyzer = SentimentIntensityAnalyzer()
    return analyzer


def get_sentiment_scores(txn_df: pd.DataFrame) -> pd.DataFrame:
    """Get sentiment scores"""
    sent_obj = "Z_sentiment_object"
    sent_neg_col = "neg"
    sent_compound_col = "compound"

    dc.logger.info("Initiating Sentiment Analysis Model")
    try:
        sent_analyzer = load_sentiment_model()
    except Exception as e:
        dc.logger.error(e)
        return None
    txn_df[sent_obj] = txn_df[dc.txn_description].apply(
        lambda x: sent_analyzer.polarity_scores(x)
    )
    txn_df[sent_neg_col] = txn_df[sent_obj].apply(lambda x: x["neg"])
    txn_df[sent_compound_col] = txn_df[sent_obj].apply(lambda x: x["compound"])
    txn_df = txn_df.drop([sent_obj], axis=1)

    return txn_df


def percentile(n: int) -> Callable:
    def percentile_(x):
        return np.percentile(x, n)

    return percentile_


def get_time_features(
    final_edges: pd.DataFrame, final_set: pd.DataFrame
) -> Tuple[int, pd.DataFrame]:
    try:
        final_set["date"] = final_set["tx_date"].dt.date
        final_set["max_trans_in_day"] = 1

        final_set_date_l = (
            final_set[
                ["sender_id", "receiver_id", "date", "max_trans_in_day", "year_month"]
            ]
            .groupby(["sender_id", "receiver_id", "date", "year_month"])
            .agg("count")
        )
        final_set_date_l.reset_index(inplace=True)

        final_set_date = (
            final_set_date_l.groupby(["sender_id", "receiver_id", "year_month"])
            .agg({"date": "nunique", "max_trans_in_day": max})
            .reset_index()
        )

        final_set_date["day_between_max_min"] = (
            final_set_date_l[["sender_id", "receiver_id", "date", "year_month"]]
            .groupby(["sender_id", "receiver_id", "year_month"])["date"]
            .transform(lambda x: (x.max() - x.min()).days)
        )
        final_edges = pd.merge(
            final_edges, final_set_date, on=["sender_id", "receiver_id", "year_month"]
        )
    except Exception as E:
        dc.logger.error("Error in Date-Time Feature Engineering")
        dc.logger.error(str(E))

    final_edges.drop_duplicates(inplace=True)

    return 0, final_edges


def get_edge_features(
    txn_df: pd.DataFrame, final_set: pd.DataFrame
) -> Tuple[int, pd.DataFrame, pd.DataFrame]:
    final_edges = pd.DataFrame()
    final_set = pd.merge(
        txn_df[[dc.edge_feat_source, dc.edge_feat_target, dc.txn_data_ind_col]],
        final_set,
    )  # HC
    dict_fun = dict(zip(final_set.columns, [["mean"]] * len(final_set.columns)))
    drop_features = set(
        [
            "tx_date",
            "sender_id",
            "receiver_id",
            "Z_receipt_number",
            "tx_description",
            "processed_description",
        ]
    ).intersection(final_set.columns)

    for _ in drop_features:
        dict_fun.pop(_)

    for key, value in dict_fun.items():
        if len(key) == 3:
            dict_fun[key] = "max"
        elif key in [
            "length_transaction",
            "longest_word",
            "number_words",
            "words_prop_length",
            "neg",
            "sent_neg",
        ]:
            dict_fun[key] = ["min", "max", "median"]
        elif key in ["anger", "fear", "joy", "love", "neutral", "sadness", "surprise"]:
            dict_fun[key] = "sum"
        elif key in [
            "toxicity",
            "severe_toxicity",
            "obscene",
            "identity_attack",
            "insult",
            "threat",
            "sexual_explicit",
        ]:
            dict_fun[key] = [percentile(75)]

    if dc.count_features:
        dict_fun["num_trans"] = "sum"
        final_set["num_trans"] = 1

    final_set["tx_date"] = pd.to_datetime(final_set["tx_date"])
    final_set["year_month"] = (
        final_set["tx_date"].dt.year * 100 + final_set["tx_date"].dt.month
    )

    final_edges_features = (
        final_set.groupby(["sender_id", "receiver_id", "year_month"])
        .agg(dict_fun)
        .reset_index()
    )
    final_edges_features.columns = final_edges_features.columns.map("_".join).str.strip(
        "_"
    )
    final_edges = final_edges_features
    dc.logger.info(f"Edge feature df shape: {final_edges_features.shape}")

    return 0, final_edges, final_set


def normalize_features(final_edges: pd.DataFrame) -> pd.DataFrame:
    def emo_norm(row, column):
        return row[column] / row["num_trans_sum"]

    filter_col = [
        "sender_id",
        "receiver_id",
        "year_month",
        "processed_description",
        "joy_max",
    ]

    final_edges["anger_sum"] = final_edges.apply(
        lambda x: emo_norm(x, "anger_sum"), axis=1
    )
    final_edges["fear_sum"] = final_edges.apply(
        lambda x: emo_norm(x, "fear_sum"), axis=1
    )
    final_edges["joy_max"] = final_edges.apply(lambda x: emo_norm(x, "joy_max"), axis=1)
    final_edges["love_sum"] = final_edges.apply(
        lambda x: emo_norm(x, "love_sum"), axis=1
    )
    final_edges["neutral_sum"] = final_edges.apply(
        lambda x: emo_norm(x, "neutral_sum"), axis=1
    )
    final_edges["sadness_sum"] = final_edges.apply(
        lambda x: emo_norm(x, "sadness_sum"), axis=1
    )
    final_edges["surprise_sum"] = final_edges.apply(
        lambda x: emo_norm(x, "surprise_sum"), axis=1
    )
    # if not score:
    try:
        final_edges = final_edges.apply(
            lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
            if x.name not in filter_col
            else x
        )
    except Exception as e:
        print(f"Failed to normalize : {str(e)}")

    dc.logger.info(f"final edge features before filtering: {final_edges.shape}")
    dc.logger.info("Generating Final Feature Set")

    return final_edges
