"""
Basic Perceptron implementation.
"""

# System Imports.
import numpy, pandas

# User Class Imports.
from resources import logging
import neural_net


# Initialize logging.
logger = logging.get_logger(__name__)


# Load in CSV.
housing_data = pandas.read_csv('./Documents/boston_housing.csv')

# Randomize arrangement of data to get more useful samples.
randomized_data = housing_data.iloc[numpy.random.permutation(len(housing_data))]

# Separate columns into "features" and "targets".
features = randomized_data.loc[:, randomized_data.columns != 'medv']
targets = randomized_data['medv']

# Get target median.
target_median = targets.median()


# Start and use perceptron.
logger.info('Starting Perceptron\n')
perceptron = neural_net.Perceptron()
perceptron.initialize_weights(features)
perceptron.get_target_median(target_median)

# Grab training/testing sets.
training_data = randomized_data[0:450]
testing_features = features[451:506]
testing_targets = targets[451:506]

# Train and predict appropriate sets.
perceptron.train(training_data, 10)
perceptron.predict(testing_features.values, target=testing_targets.values)

logger.info('Exiting program.')
