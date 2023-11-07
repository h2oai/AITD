import pandas as pd
import h2o
import torch
import sys
from multiprocessing import Process, Queue
from typing import Any

from src import config, simpletext, publicmodels, sentiment
from src import process_transactions as pt
from src.process_transactions import *

def get_emotion_feature(txn_df: pd.DataFrame, q: Any = None) -> pd.DataFrame:
    config.logger.info(f"getting emotion features")
    exit_code = 0
    emo_df = pd.DataFrame()
    try:
        emo_df = publicmodels.get_emotion_model_results(txn_df)
    except Exception as e:
        config.logger.error(f"Failed to get emotion features {str(e)}")
        exit_code = 1
    if q:
        q.put([exit_code, emo_df])
    return exit_code, emo_df


def get_tox_feature(txn_df: pd.DataFrame, q: Any = None) -> pd.DataFrame:
    config.logger.info(f"getting tox features")
    exit_code = 0
    tox_df = pd.DataFrame()
    try:
        pm = publicmodels.public_models(txn_df)
        tox_df = pm.detoxify_model()
    except Exception as e:
        config.logger.error(f"Failed to get tox feature {str(e)}")
        exit_code = 1
    if q:
        q.put([exit_code, tox_df])
    return exit_code, tox_df


def get_sentiment_feature(txn_df: pd.DataFrame, q: Any = None) -> pd.DataFrame:
    config.logger.info(f"getting sentiment features")
    exit_code = 0
    sentiment_df = pd.DataFrame()
    try:
        sentiment_df = sentiment.get_sentiment_scores(txn_df)
    except Exception as e:
        config.logger.error(f"Failed to get sentiment feature {str(e)}")
        exit_code = 1
    if q:
        q.put([exit_code, sentiment_df])

    return exit_code, sentiment_df


def get_text_features(txn_df: pd.DataFrame, q: Any = None) -> pd.DataFrame:
    config.logger.info(f"getting text features")
    exit_code = 0
    text_feat_df = pd.DataFrame()
    try:
        status, text_feat_df = simpletext.get_simple_text_features(txn_df)
        if status != 0:
            raise SystemExit()
    except Exception as e:
        config.logger.error(f"Failed to get text feature {str(e)}")
        exit_code = 1
    if q:
        q.put([exit_code, text_feat_df])
    return exit_code, text_feat_df


def transform(data: pd.DataFrame, score_month = None, score_year = None, lag = 2) -> pd.DataFrame:
    """
    Transforms the input data
    added columns:
        - emotion features
        - toxicity features
        - sentiment features
        - text features
    Convert the data from long to wide format

    Args:
        data(pd.DataFrame):
            input dataframe, consists of the columns: tx_description, sender_id, receiver_id, tx_date, amount

    Returns:
        pd.DataFrame
    """

    # preprocess_txn_data
    config.logger.info(f"transforming features")
    txn_df = pt.pre_process_txn(data)
    pt.sanity_check(txn_df, score_month = score_month, score_year = score_year, lag = lag)
    # print(txn_df)
    if torch.cuda.is_available():
        ec1, emo_df = get_emotion_feature(txn_df.copy())
        ec2, tox_df = get_tox_feature(txn_df.copy())
        ec3, sentiment_df = get_sentiment_feature(txn_df.copy())
        ec4, text_feat_df = get_text_features(txn_df.copy())
    #elif SEQ_MODE:
    else:
        q1, q2, q3, q4 = Queue(), Queue(), Queue(), Queue()
        p1 = Process(target=get_emotion_feature, args=(txn_df.copy(), q1))
        p2 = Process(target=get_tox_feature, args=(txn_df.copy(), q2))
        p3 = Process(target=get_sentiment_feature, args=(txn_df.copy(), q3))
        p4 = Process(target=get_text_features, args=(txn_df.copy(), q4))
        [p.start() for p in [p1, p2, p3, p4]]
        ec1, emo_df = q1.get()
        ec2, tox_df = q2.get()
        ec3, sentiment_df = q3.get()
        ec4, text_feat_df = q4.get()
        [sys.exit(1) for i in [ec1, ec2, ec3, ec4] if i == 1]
        [p.join() for p in [p1, p2, p3, p4]]

    # merge_models_text_features
    [
        sys.exit(
            f"Error: error(s) in feature generation step(s), see {config.LOG_FILE_DIR} folder for more detail"
        )
        for i in [ec1, ec2, ec3, ec4]
        if i == 1
    ]
    prefinal_set = merge_tox_amo_scores(tox_df, emo_df, text_feat_df, sentiment_df)

    # get_edgetime_features
    edge_df = pd.DataFrame()
    _, edge_df, final_set = get_edge_features(txn_df.copy(), prefinal_set)
    final_edges = pd.DataFrame()
    _, final_edges = get_time_features(edge_df, final_set)

    # generate_final_featureset
    feature_df = gen_features(final_edges, txn_df.copy(), lag=lag, score_month=score_month, score_year=score_year)
    config.logger.info(f"finished transforming features")
    return feature_df


def score(data: pd.DataFrame, model_loc: str = "models/AITD_Model.zip") -> pd.DataFrame:
    """
    scores the input data using h2o model

    Args:
        data(pd.DataFrame):
            input dataframe, consists of the columns: tx_description, sender_id, receiver_id, tx_date, amount
        model_loc(str):
            path of the h2o model

    Returns:
        pd.DataFrame
    """
    config.logger.info(f"scoring transaction abuse")
    h2o.init()
    
    # Model Defs
    data = data.fillna(0)
    imported_model = h2o.import_mojo(model_loc)
    list_of_features = list(imported_model._model_json["output"]["names"])[:-1]

    try:
        # Set-up Score Data
        X = h2o.H2OFrame(data)
        x_score_h2o = X[list_of_features]
    except:
        missing = set(list_of_features) - set(list(data.columns))
        print("Failed - Missing columns: " + str(missing) + ". Please check you are scoring the correct model for your historical data length. " )
        return
    
    
    predictions = imported_model.predict(x_score_h2o)

    # Format Outputs
    predictions["sender_id"] = X["sender_id"]
    predictions["receiver_id"] = X["receiver_id"]
    predictions = predictions[["sender_id", "receiver_id", "p0", "p1"]]
    predictions.columns = [
        "sender_id",
        "receiver_id",
        "probability_non_abuse",
        "probability_abuse",
    ]

    # Convert To DataFrame
    predictions_out = predictions.as_data_frame()
    predictions_out = predictions_out.sort_values("probability_abuse", ascending=False)

    h2o.cluster().shutdown()
    config.logger.info(f"finished scoring transaction abuse")

    return predictions_out
