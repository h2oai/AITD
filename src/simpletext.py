import string
import pandas as pd
from pandarallel import pandarallel
from string import punctuation
from typing import Tuple

# local imports
from src.config import *


class simple_text_features:
    def __init__(
        self, transaction_descriptions: pd.DataFrame, workers: int = 1
    ) -> None:
        """
        Initialise this class
        transaction_descriptions: Data Frame with features
        workers: number of cores to use for paralellisation (1 default)
        """
        self.is_parallel = workers != 1
        if self.is_parallel:
            pandarallel.initialize(
                nb_workers=workers, progress_bar=False, use_memory_fs=False
            )
        self.transaction_descriptions = transaction_descriptions

    """
    Simple Text Features calculation for all transaction descriptions in a Data Frame
    """

    # Length of Transaction Description
    def length(self) -> pd.Series:
        if self.is_parallel:
            return self.transaction_descriptions.parallel_apply(len)
        return self.transaction_descriptions.apply(len)

    # Whether the transaction has upper case and lower case
    def mixed_case(self) -> pd.Series:
        def mixed_case_individual(row):
            return int(not row[1:].islower() and not row.isupper())

        if self.is_parallel:
            return self.transaction_descriptions.parallel_apply(mixed_case_individual)
        return self.transaction_descriptions.apply(mixed_case_individual)

    # Whether the transaction is all lower case
    def lower(self) -> pd.Series:
        def lower_individual(row):
            return int(row.islower())

        if self.is_parallel:
            return self.transaction_descriptions.parallel_apply(lower_individual)
        return self.transaction_descriptions.apply(lower_individual)

    # Whether the transaction is all upper case
    def upper(self) -> pd.Series:
        def upper_individual(row):
            return int(row.isupper())

        if self.is_parallel:
            return self.transaction_descriptions.parallel_apply(upper_individual)
        return self.transaction_descriptions.apply(upper_individual)

    # Number of words in a transaction description
    def numWords(self) -> pd.Series:
        def num_words_individual(row):
            return len(row.split())

        if self.is_parallel:
            return self.transaction_descriptions.parallel_apply(num_words_individual)
        return self.transaction_descriptions.apply(num_words_individual)

    # Whether the transaction contains punctuation
    def punctuationsjd(self) -> pd.Series:
        def punctuation_individual(row):
            return int(bool(any(p in row for p in punctuation)))

        if self.is_parallel:
            return self.transaction_descriptions.parallel_apply(punctuation_individual)
        return self.transaction_descriptions.apply(punctuation_individual)

    # Whether the transaction is only made up of punctuation or numbers
    def all_punctuation_or_number(self) -> pd.Series:
        def all_punc_num_individual(row):
            return int(
                bool(
                    all(j.isdigit() or j in string.punctuation for j in i) for i in row
                )
            )

        if self.is_parallel:
            return self.transaction_descriptions.parallel_apply(all_punc_num_individual)
        return self.transaction_descriptions.apply(all_punc_num_individual)

    # Whether the transaction contains
    def number_contains(self) -> pd.Series:
        def number_contains_individual(row):
            return int(bool(any(j.isdigit() for j in i) for i in row))

        if self.is_parallel:
            return self.transaction_descriptions.parallel_apply(
                number_contains_individual
            )
        return self.transaction_descriptions.apply(number_contains_individual)

    # The length of the longest word in the transaction
    def find_longest_word(self) -> pd.Series:
        def find_longest_word_individual(row):
            word_list = row.split()
            longest_size = 0
            for word in word_list:
                if len(word) > longest_size:
                    longest_size = len(word)
            return longest_size

        if self.is_parallel:
            return self.transaction_descriptions.parallel_apply(
                find_longest_word_individual
            )
        return self.transaction_descriptions.apply(find_longest_word_individual)

    # The proportion of words to the length of the transaction
    def proportion_word_length(self) -> pd.Series:
        def num_words_individual(row):
            return len(row.split())

        def proportion_word_length_individual(row):
            if num_words_individual(row) != 0 and len(row) != 0:
                return num_words_individual(row) / len(row)
            else:
                return 0

        if self.is_parallel:
            return self.transaction_descriptions.parallel_apply(
                proportion_word_length_individual
            )
        return self.transaction_descriptions.apply(proportion_word_length_individual)


def get_simple_text_features(transactions) -> Tuple[int, pd.DataFrame]:
    ret_val = 0
    try:
        st = simple_text_features(transactions.tx_description)
        transactions["length_transaction"] = st.length()
        transactions["mixedcase"] = st.mixed_case()
        transactions["lowercase"] = st.lower()
        transactions["upper_case"] = st.upper()
        transactions["number_words"] = st.numWords()
        transactions["punctuation_found"] = st.punctuationsjd()
        transactions["all_punctuation_or_number"] = st.all_punctuation_or_number()
        transactions["longest_word"] = st.find_longest_word()
        transactions["number_contains"] = st.number_contains()
        transactions["words_prop_length"] = st.proportion_word_length()
        transactions["num_trans"] = 1
    except Exception as E:
        logger.error("Error in Simple Text Features")
        logger.error(str(E))
        return -1, transactions

    return ret_val, transactions
