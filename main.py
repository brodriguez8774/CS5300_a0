"""
Basic Perceptron implementation.
"""

# System Imports.
from matplotlib import patches, pyplot
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


logger.info('Starting Perceptron\n')

# Grab training/testing sets.
training_data = randomized_data[0:450]
testing_features = features[451:506]
testing_targets = targets[451:506]

# Train and predict appropriate sets.
training_results = []
prediction_results = []
index = 0

while index < 500:
    logger.info('Starting perceptron #{0}'.format(index))
    perceptron = neural_net.Perceptron()
    perceptron.initialize_weights(features)
    perceptron.get_target_median(target_median)
    training_results.append(perceptron.train(training_data, 10))
    prediction_results.append(perceptron.predict(testing_features.values, target=testing_targets.values))
    index += 1


logger.info('')
logger.info('Training Results:')
logger.info('{0}'.format(training_results))
logger.info('')
logger.info('Prediction Results:')
logger.info('{0}'.format(prediction_results))
logger.info('')


training_numpy_array = numpy.asarray(training_results)
prediction_numpy_array = numpy.asarray(prediction_results)
x = []
y = []

for result in training_results:
    # pyplot.plot(result[1], result[0], 'bo')
    pyplot.scatter(result[1], result[0], alpha=0.10, c='b')
    x.append(result[1])
    y.append(result[0])

# Plot labels.
pyplot.title('Perceptron Training Results after 500 Runs')
pyplot.xlabel('Total Iterations')
pyplot.ylabel('Best Accuracy')

# Create average line.
y_mean = [numpy.mean(y) for i in y]
pyplot.plot(x, y_mean, 'g', label='Mean')
label_1 = patches.Patch(color='g', label='Mean')
pyplot.legend(handles=[label_1])

pyplot.show()


index = 0
x = []
y = []
for result in prediction_results:
    # pyplot.plot(training_results[index][1], result, 'bo')
    pyplot.scatter(training_results[index][1], result, alpha=0.10, c='b')
    x.append(training_results[index][1])
    y.append(result)
    index += 1

# Plot labels.
pyplot.title('Perceptron Prediction Results after 500 Runs')
pyplot.xlabel('Total Training Iterations')
pyplot.ylabel('Accuracy')

# Create average line.
y_mean = [numpy.mean(y) for i in y]
pyplot.plot(x, y_mean, 'g', label='Mean')
label_1 = patches.Patch(color='g', label='Mean')
pyplot.legend(handles=[label_1])

pyplot.show()


logger.info('Exiting program.')
