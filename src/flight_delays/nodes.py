"""
This is a boilerplate pipeline
generated using Kedro 0.18.3
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, types
from pyspark.sql.functions import col, upper, trim, length, split


def select_cols(df: DataFrame, params: Dict):
    df = df.select(params["l_cols"])
    return df


def clean_data(df: DataFrame, params: Dict):
    logging.info(f"Count before data cleaning is {df.count()}")
    df = (
        df.dropna()
        .withColumn("Airline", trim(upper("Reporting_Airline")))
        .where(length(col("Airline")) == 2)
        .drop("Reporting_Airline")
    )
    logging.info(f"Count AFTER data cleaning is {df.count()}")

    return df


def feature_extraction(df: DataFrame, params: Dict):
    df_clean = (
        df.withColumn("DepHour", (col("DepTime") / 100).cast(types.IntegerType()))
        .drop("DepTime")
        .withColumn(
            "DepMonth",
            split(col("FlightDate"), "-").getItem(1).cast(types.IntegerType()),
        )
        .withColumn(
            "DepYear",
            split(col("FlightDate"), "-").getItem(0).cast(types.IntegerType()),
        )
        .drop("FlightDate")
    )

    return df_clean


def split_data(data: DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """

    # Split to training and testing data
    data = data[parameters["features"]]
    data_train, data_test = data.randomSplit(
        weights=[parameters["train_fraction"], 1 - parameters["train_fraction"]]
    )

    X_train = data_train.drop(parameters["target_column"])
    X_test = data_test.drop(parameters["target_column"])
    y_train = data_train.select(parameters["target_column"])
    y_test = data_test.select(parameters["target_column"])

    return X_train, X_test, y_train, y_test


def report_accuracy(y_pred: pd.Series, y_test: pd.Series):
    """Calculates and logs the accuracy.

    Args:
        y_pred: Predicted target.
        y_test: True target.
    """
    accuracy = (y_pred == y_test).sum() / len(y_test)
    logger = logging.getLogger(__name__)
    logger.info("Model has an accuracy of %.3f on test data.", accuracy)
