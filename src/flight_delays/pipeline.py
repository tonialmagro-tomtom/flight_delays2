"""
This is a boilerplate pipeline
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_data, feature_extraction, select_cols, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=select_cols,
                inputs=["flights", "parameters"],
                outputs="flights_sel",
                name="select_cols",
            ),
            node(
                func=clean_data,
                inputs=["flights_sel", "parameters"],
                outputs="flights_cleaned",
                name="clean_data",
            ),
            node(
                func=feature_extraction,
                inputs=["flights_cleaned", "parameters"],
                outputs="flights_features",
                name="feature_extraction",
            ),
            node(
                func=split_data,
                inputs=["flights_features", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data",
            ),
        ]
    )
