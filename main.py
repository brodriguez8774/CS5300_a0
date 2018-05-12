"""
Basic Perceptron implementation.
"""

# System Imports.
import numpy, pandas

# User Class Imports.
from resources import logging

# Initialize logging.
logger = logging.get_logger(__name__)


# Load in CSV.
housing_data = pandas.read_csv('./Documents/boston_housing.csv')
logger.info('CSV Data:\n{0}\n'.format(housing_data.head()))

# Randomize arrangement of data to get more useful samples.
randomized_data = housing_data.iloc[numpy.random.permutation(len(housing_data))]
logger.info('Randomized Data:\n{0}\n'.format(randomized_data.head()))

# Separate columns into "features" and "targets".
features = randomized_data.loc[:, randomized_data.columns != 'medv']
targets = randomized_data['medv']
logger.info('Separated column data:')
logger.info(features.head())
logger.info(targets.head())


class Perceptron(object):
    def __init__(self):
        """
        Create and initialize a new Perceptron object.
        """
        pass

    def predict(self, x):
        """
        Predict the class of sample x. (Forward pass)
        """
        pass

    def _delta(self, y_hat, y):
        """
        Given predictions y_hat and targets y, calculate the weight update delta.
        """
        pass

    def _update_weights(delta):
        """
        Update the weights by delta.
        """
        pass

    def _train_step(self, x, y):
        """
        Perform one training step:
            - Predict.
            - Calculate delta.
            - Update weights.
        Returns the predictions y_hat.
        """
        y_hat = self.predict(x)
        delta = self._delta(y_hat, y)
        self._update_weights(delta)
        return y_hat

    def train(self, train_x, train_y, num_steps):
        """
        Train the perceptron, performing num_steps weight updates.
        """
        pass


logger.info('Exiting program.\n\n')
