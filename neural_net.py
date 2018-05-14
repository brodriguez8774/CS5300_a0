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
                indiv_set += column * self.weights[col_index]
                col_index += 1

            if indiv_set > self.target_median:
                full_set.append(1)
            else:
                full_set.append(0)
            instance_index += 1

        if target is not None:
            # Calculate the error margin.
            index = 0
            error_count = 0
            while index < len(full_set):
                if target[index] > self.target_median:
                    target_value = 0
                else:
                    target_value = 1
                if full_set[index] != target_value:
                    error_count += 1
                index += 1
            logger.info('Testing: {0} predictions incorrect out of {1}. Error margin of {2}%'
                        .format(error_count, len(full_set), (error_count / len(full_set) * 100)))

        return full_set

    def train(self, train_x, train_y, num_steps):
        """
        Train the perceptron, performing num_steps weight updates.
        :param train_x: Feature training data.
        :param train_y: Result training data.
        :param num_steps: Number of iterations to update weights with.
        :return:
        """
        logger.info('Number of Steps: {0}'.format(num_steps))
        # First determine number of items per training set.
        total_set_count = len(train_x)
        sets_per_step = total_set_count / num_steps
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
            # logger.info('Set from {0} to {1}'.format(curr_index, math.floor(high_index)))
            sample_features = train_x[curr_index : math.floor(high_index)]
            sample_targets = train_y[curr_index : math.floor(high_index)]
            results = self._train_step(sample_features, sample_targets)

            curr_index = math.ceil(high_index)

            # Calculate the error margin.
            index = 0
            error_count = 0
            while index < len(results):
                if sample_targets[index] > self.target_median:
                    target_value = 0
                else:
                    target_value = 1
                if results[index] != target_value:
                    error_count += 1
                index += 1
            logger.info('Training: {0} predictions incorrect out of {1}. Error margin of {2}%'
                        .format(error_count, len(results), (error_count / len(results) * 100)))


    def _train_step(self, x, y):
        """
        Perform one training step:
            - Predict.
            - Calculate delta.
            - Update weights.
        :return: The predictions y_hat.
        """
        y_hat = self.predict(x)
        delta = self._delta(y_hat, y)
        self._update_weights(delta)
        return y_hat

    def _delta(self, results, target):
        """
        Given prediction results (y_hat) and targets (y), calculate the weight update delta.
        Delta is equal to: (result - target)^2
        """
        index = 0
        delta = 0
        while index < len(results):
            if target[index] > self.target_median:
                target_median = 0
            else:
                target_median = 1
            delta += ((results[index] - target_median) ** 2)
            index += 1
        # delta = (delta / len(results))
        # logger.info('Delta: {0}'.format(delta))
        return delta

    def _update_weights(self, delta):
        """
        Update the weights by delta.
        """
        # TODO: This seems way to simple to be correct.
        # TODO: How do you update each weight individually? There's only one delta.
        for weight in self.weights:
            weight += delta
