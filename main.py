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
# logger.info('CSV Data:\n{0}\n'.format(housing_data.head()))

# Randomize arrangement of data to get more useful samples.
randomized_data = housing_data.iloc[numpy.random.permutation(len(housing_data))]
# logger.info('Randomized Data:\n{0}\n'.format(randomized_data.head()))

# Separate columns into "features" and "targets".
features = randomized_data.loc[:, randomized_data.columns != 'medv']
targets = randomized_data['medv']
# logger.info('Separated column data:')
# logger.info(features.head())
# logger.info(targets.head())

# Get target median.
target_median = targets.median()
# logger.info('Target Median: {0}'.format(target_median))

# Experimenting with columns.
# logger.info('Columns: ')
# logger.info(randomized_data.columns)
# logger.info('Col Length: {0}\n'.format(len(randomized_data.columns)))


# # To simplify things, only work with two features for now.
# features = features.loc[:, features.columns != 'zn']
# features = features.loc[:, features.columns != 'indus']
# features = features.loc[:, features.columns != 'chas']
# features = features.loc[:, features.columns != 'nox']
# features = features.loc[:, features.columns != 'rm']
# features = features.loc[:, features.columns != 'dis']
# features = features.loc[:, features.columns != 'rad']
# features = features.loc[:, features.columns != 'tax']
# features = features.loc[:, features.columns != 'ptratio']
# features = features.loc[:, features.columns != 'b']
# features = features.loc[:, features.columns != 'lstat']
# logger.info(features.columns)





# Start and use perceptron.
logger.info('Starting Perceptron\n')
perceptron = neural_net.Perceptron()
perceptron.initialize_weights(features)
perceptron.get_target_median(target_median)

# Grab training/testing sets.
training_features = features[0:450]
training_targets = targets[0:450]
testing_features = features[451:506]
testing_targets = targets[451:506]

perceptron.train(training_features.values, training_targets.values, 10)
perceptron.predict(testing_features.values, testing_targets.values)
perceptron.predict(testing_features.values)


logger.info('Exiting program.\n\n')
