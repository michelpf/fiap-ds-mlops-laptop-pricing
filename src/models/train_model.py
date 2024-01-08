# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from dotenv import find_dotenv, load_dotenv

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

import matplotlib.pyplot as plt

from dvclive import Live

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    with Live() as live:
        logger = logging.getLogger(__name__)
        logger.info('Load the processed data.')
        
        df = pd.read_csv(input_filepath)
        
        features = list(df.columns)
        features.remove("price")
        
        logger.info('Number of features: ' + str(len(features)))
        logger.info('Features: ' + str(features))

        X = df[features]
        y = df["price"]

        logger.info('Split the data for training and testing')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        logger.info('Model is based on Linear regression.')
        model = LinearRegression(fit_intercept=True)

        logger.info('Starting model training.')
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        logger.info('Mean Squared Error: ' + str(mse))
        logger.info('Mean Absolute Error: ' + str(mae))
        logger.info('R2: ' + str(r2))

        live.log_metric("mse", mse, timestamp=True, plot=True)
        live.log_metric("mae", mae, timestamp=True, plot=True)
        live.log_metric("r2", r2, timestamp=True, plot=True)
        
        logger.info('Saving figure of results.')
        plt.scatter(y_test, predictions)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Scatter Plot of True vs Predicted Values')
        plt.savefig("./reports/figures/true_vs_predicted.png",dpi=80)

        logger.info('Model serialization.')
        joblib.dump(model, output_filepath)

        live.log_artifact(output_filepath, type="model", name="laptop-pricing", desc="Model to predict laptop pricing.", labels=["regression"])

        logger.info('Process finished.')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
