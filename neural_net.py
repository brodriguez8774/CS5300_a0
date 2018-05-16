"""
Neural Net logic.
"""

# System Imports.
import math

# User Class Imports.
from resources import logging


# Initialize logging.
logger = logging.get_logger(__name__)


class Perceptron(object):
    """
    Perceptron object. Acts as neural net.
    """

    def __init__(self):
        """
        Create and initialize a new Perceptron object.
        """
        self.data = None
        self.weights = None
        self.bias = -10
        self.target_median = None

    def initialize_weights(self, data):
        self.weights = []
        for column in data.columns:
            self.weights.append(1)
        logger.info('Weights: {0}'.format(self.weights))

    def get_target_median(self, target_median):
        self.target_median = target_median

    def predict(self, sample, target=None):
        """
        Predict the class of sample (x). (Forward pass)
        """
        # logger.info('Starting prediction.')
        instance_index = 0
        full_set = []

        # Iterate through all instances.
        while instance_index != len(sample):
            # logger.info('Current Set: {0}'.format(sample[instance_index]))
            col_index = 0
            indiv_set = self.bias

            # Iterate through all columns in instance. Should result with single value for instance.
            for column in sample[instance_index]:
                # logger.info('Current Col: {0}'.format(column))
                indiv_set += (column * self.weights[col_index])
                col_index += 1

            full_set.append(indiv_set)
            instance_index += 1

        # Calculate the error margin if not a test prediction.
        if target is not None:
            binary_results = self._convert_to_binary_result(full_set)
            binary_targets = self._convert_to_binary_result(target)
            self._determine_error_margin(binary_results, binary_targets)


        return full_set

    def train(self, train_x, train_y, num_steps):
        """
        Train the perceptron, performing num_steps weight updates.
        :param train_x: Feature training data.
        :param train_y: Result training data.
        :param num_steps: Number of iterations to update weights with.
        :return:
        """
        # First determine number of items per training set.
        total_set_count = len(train_x)
        sets_per_step = total_set_count / num_steps
        logger.info('Number of Steps: {0}'.format(num_steps))
        logger.info('Total Set Count: {0}   Sets per Step: {1}'.format(total_set_count, sets_per_step))

        # Iterate through sets and train perceptron accordingly.
        curr_index = 0
        high_index = 0
        while high_index < (total_set_count - 1):
            high_index += sets_per_step

            # Check that we don't exceed the max index.
            if high_index >= total_set_count:
                high_index = (total_set_count - 1)

            # Train on the given sets.
            error_margin = 100
            sample_features = train_x[curr_index: math.floor(high_index)]
            sample_targets = train_y[curr_index: math.floor(high_index)]
            binary_targets = self._convert_to_binary_result(sample_targets)

            while error_margin > 30:
                results = self._train_step(sample_features, sample_targets)

                binary_results = self._convert_to_binary_result(results)

                # Calculate the error margin.
                error_margin = self._determine_error_margin(binary_results, binary_targets, training=True)

            curr_index = math.ceil(high_index)

    def _train_step(self, x, y):
        """
        Perform one training step:
            - Predict.
            - Calculate delta.
            - Update weights.
        :return: The predictions y_hat.
        """
        y_hat = self.predict(x)
        logger.info('Training Step Data:')
        logger.info('Feature Calcs: {0}'.format(y_hat))
        logger.info('Target Calcs:  {0}'.format(y))

        # Calculate binary of values and compare against targets.
        binary_results = self._convert_to_binary_result(y_hat)
        binary_targets = self._convert_to_binary_result(y)

        delta = self._delta(x, binary_results, binary_targets)
        logger.info('Delta: {0}'.format(delta))
        self._update_weights(delta)
        return y_hat

    def _delta(self, features, results, target):
        """
        Given prediction results (y_hat) and targets (y), calculate the weight update delta.
        """
        row_index = 0
        delta = []
        for weight in self.weights:
            delta.append(0)

        # Iterate through all rows in features.
        for set in features:
            weight_index = 0
            # Iterate through all weights in delta.
            for weight in delta:
                # logger.info('Before | Weight {0}: {1}   Result Value: {2}   Target Value: {3}'.format(weight_index, delta[weight_index], results[row_index], target[row_index]))
                delta[weight_index] += ((target[row_index] - results[row_index]) * set[weight_index])
                # logger.info('After  | Weight {0}: {1}   Result Value: {2}   Target Value: {3}'.format(weight_index, delta[weight_index], results[row_index], target[row_index]))
                weight_index += 1
            row_index += 1

        # Average out delta values.
        index = 0
        while index < len(delta):
            delta[index] = delta[index] / len(features)
            index += 1

        return delta

    def _update_weights(self, delta):
        """
        Update the weights by delta.
        """
        index = 0
        logger.info('Old Weights: {0}'.format(self.weights))
        while index < len(delta):
            self.weights[index] += delta[index]
            index += 1
        logger.info('New Weights: {0}'.format(self.weights))

    def _convert_to_binary_result(self, result_array):
        """
        Converts the given result array to 1's or 0's, based on if it's above or below median value.
        :param result_array: Array of data to convert to binary.
        :return: Binary version of array.
        """
        binary_result = []
        for result in result_array:
            if result > self.target_median:
                binary_result.append(1)
            else:
                binary_result.append(0)
        return binary_result

    def _determine_error_margin(self, results, targets, training=False):
        """
        Determines error margin of given values.
        :param results: Binary array of results.
        :param targets: Binary array of targets.
        :param training: Boolean to change output.
        :return: Current error margin.
        """
        # Calculate the error margin.
        index = 0
        error_count = 0
        while index < len(results):
            if results[index] != targets[index]:
                error_count += 1
            index += 1
        error_margin = (error_count / len(results) * 100)
        if training:
            logger.info('Training   | {0} predictions incorrect out of {1}. Error margin of {2}%'
                        .format(error_count, len(results), error_margin))
        else:
            logger.info('Predicting | {0} predictions incorrect out of {1}. Error margin of {2}%'
                        .format(error_count, len(results), error_margin))

        return error_margin
