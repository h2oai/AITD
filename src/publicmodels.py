import pandas as pd
import numpy as np
import wordninja
import re
import GPUtil
import torch
from typing import Dict

# from pandarallel import pandarallel
from transformers import pipeline

# local imports
from src import detoxify
from src import config as dc


def unicode_escape(text: str) -> str:
    try:
        return text.encode("utf-8").decode("unicode_escape")
    except Exception:
        return text


def preprocessing(text: str) -> str:
    if text == "":
        return text
    text = unicode_escape(text)
    text = re.sub(r"\d+", "", text)  # Remove numbers

    # Split long messages that have no spaces
    tokens = []
    for word in text.split(" "):
        if len(word) > 5:
            tokens += wordninja.split(word)
        else:
            tokens.append(word)

    return " ".join(x for x in tokens)


class public_models:
    # Parameter to add - batch size here.
    def __init__(
        self, transactions: pd.DataFrame, num_workers: int = dc.num_workers
    ) -> None:
        transactions_copy = transactions.copy()
        transactions_copy["processed_description"] = transactions_copy[
            "tx_description"
        ].apply(preprocessing)
        transactions_copy = transactions_copy.reset_index(drop=True)
        n_procs = max(
            int(len(transactions_copy) / dc.max_batch_size), dc.min_batch_size
        )

        self.trans_df_batches = np.array_split(transactions_copy, n_procs)

        try:
            self.if_cuda = True if torch.cuda.is_available() else False
        except Exception as e:
            dc.logger.error(e)

        dc.logger.info("Public  Models Constructor initialised successfully")

        return

    def detoxify_model(self) -> pd.DataFrame:
        """
        Parameters:

        Returns:

        """
        model_path = dc.tox_model_path

        detox = detoxify.Detoxify(
            burt_model_loc=dc.tox_model_dir, checkpoint=model_path, device="cpu"
        )
        toxicity = pd.DataFrame(
            {dc.txn_data_ind_col: [], **{cl: [] for cl in detox.class_names}}
        )
        counts = 0

        # can be parallelised using mp
        for batched_text in self.trans_df_batches:
            try:
                batched_result = detox.predict(list(batched_text.processed_description))
            except Exception as e:
                dc.logger.error(e)
                return toxicity

            dict_to_append = {
                dc.txn_data_ind_col: batched_text[dc.txn_data_ind_col],
                **batched_result,
            }
            # append result dict
            toxicity = pd.concat([toxicity, pd.DataFrame(dict_to_append)])
            # if counts % 100 == 0:
            # print(counts)
            counts += 1

        del detox
        if self.if_cuda:
            # Clearing Memory
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            GPUtil.showUtilization()
            torch.cuda.empty_cache()
            GPUtil.showUtilization()

        return toxicity

    def emotion_model(self) -> Dict:
        model_path = dc.emo_model_path

        classifier = pipeline(
            "text-classification",
            model= model_path,
            return_all_scores=False,
            binary_output=True,
            device=-1,
        )  # -1 for cpu, 1 for gpu
        emotions_results = pd.DataFrame(
            {"label": [], "score": [], dc.txn_data_ind_col: []}
        )

        counts = 0
        for batched_text in self.trans_df_batches:
            try:
                emotions_result = classifier(list(batched_text.processed_description))
                temp = pd.DataFrame(emotions_result)
                temp[dc.txn_data_ind_col] = list(batched_text[dc.txn_data_ind_col])
                emotions_results = pd.concat([emotions_results, temp])
            except Exception as e:
                dc.logger.error(e)
                return emotions_result

            if self.if_cuda:
                torch.cuda.empty_cache()
            # if counts % 100 == 0:
            #     # print(counts)
            counts += 1

        dummy = pd.get_dummies(emotions_results["label"])
        emotions = pd.concat([emotions_results, dummy], axis=1)
        emotions.drop(["label", "score"], axis=1, inplace=True)

        dc.logger.info("Calculated Emotions Model Predictions")

        return emotions


def get_tox_model_results(txn_df: pd.DataFrame) -> Dict:
    """
    Parameters:
        txn_df (pd.DataFrame): input df containing transactions
    Returns:
        op (dict): Status output containing status code, status
                   and payload df containing tox values
    """
    tox_op_df = pd.DataFrame()
    tox_df = pd.DataFrame()
    op = dc.return_op

    pm = public_models(
        txn_df, model_dir=dc.tox_model_input_dir, num_workers=dc.num_workers
    )
    try:
        dc.logger.info("Initiating Toxicity Model for Feature Engineering")
        tox_op_df = pm.detoxify_model()
        dc.logger.info(
            "Toxicity df shape: {op['payload'].shape} with ret_val {op['status_code']}"
        )
    except Exception as e:
        dc.logger.error(e)
        op["status"] = str(e)
        op["status_code"] = "E604"
        op["payload"] = tox_df
        return op

    if tox_op_df["status_code"] != 0:
        return tox_op_df

    tox_df = tox_op_df["payload"]

    return op


def get_emotion_model_results(txn_df: pd.DataFrame) -> Dict:
    """
    Parameters:
        txn_df (pd.DataFrame): input df containing transactions
    Returns:
        op (dict): Status output containing status code, status
                   and payload df containing tox values
    """
    emo_df = pd.DataFrame()

    pm = public_models(txn_df, num_workers=dc.num_workers)
    try:
        dc.logger.info("Initiating Emotions Model for Feature Engineering")
        emo_df = pm.emotion_model()
        for col in dc.emo_list:
            if col not in emo_df.columns:
                emo_df[col] = 0
        dc.logger.info(f"Emotions df shape: {emo_df.shape}")
    except Exception as e:
        dc.logger.error("Error in Emotions Model")
        dc.logger.error(str(e))

    return emo_df
