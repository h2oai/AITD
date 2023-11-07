import logging
import logging.handlers
from pathlib import Path

LOGDIR = "logs"
LOGFILE_NAME = "aitd.log"

logging.basicConfig(filename=f"{LOGDIR}/{LOGFILE_NAME}",
                    filemode='a',
                    format="[%(asctime)s - %(name)s - %(levelname)s - %(filename)s->%(funcName)s:%(lineno)s] -  %(message)s",
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger('root')

SEQ_MODE = True

# Data Configurations
edge_feat_source = "sender_id"
edge_feat_target = "receiver_id"
txn_data_ind_col = "Z_receipt_number"  # Index Column
txn_data_amount_col = "amount"  # Transaction Value Column Name
txn_description = "tx_description"  # Transaction Description Column Name


# These are the features that are used for lag and reciprocal features
reference_features = [
    "sender_id",
    "receiver_id",
    "num_trans_sum",
    "anger_sum",
    "fear_sum",
    "joy_max",
    "love_sum",
    "neutral_sum",
    "sadness_sum",
    "surprise_sum",
    "toxicity_percentile",
    "severe_toxicity_percentile",
    "obscene_percentile",
    "identity_attack_percentile",
    "insult_percentile",
    "threat_percentile",
    "sexual_explicit_percentile",
    "amount_mean",
    "length_transaction_median",
    "longest_word_median",
    "date",
    "day_between_max_min",
]

# model configs
# emo_model_path = "models/transformers/distilbert-base-uncased-emotion/"
emo_model_path = "bhadresh-savani/distilbert-base-uncased-emotion"
emo_list = ["anger", "fear", "joy", "love", "neutral", "sadness", "surprise"]
tox_model_dir = "models/toxicity_model/transformers/"
tox_model_path = "models/toxicity_model/toxic_bias-4e693588.ckpt"

# base model related data
model_base_path = "models"

# save model results
save_model_results = 1  # 1 => Save and overwrite  2 => Save as a new file

count_features = True
# flag to merge results of emotion and toxicity models
merge_tox_emo_models = True

# dir where model binaries are stored
model_dir = "model_binaries"

# num workers for parallel processing
num_workers = 30

# Cuda device for GPU
device = 1

# min batch size for parallel processing
min_batch_size = 10

# max batch size for parallel processing
max_batch_size = 100
