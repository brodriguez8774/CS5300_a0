"""
Neural Net logic.
"""

# System Imports.
import math, numpy

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
        self.total_errors = None

    def initialize_weights(self, data):
        self.weights = []
        for column in data.columns:
            self.weights.append(1)

    def get_target_median(self, target_median):
        self.target_median = target_median

    def predict(self, sample, target=None):
        """
        Predict the class of sample (x).
        :param sample: Sample data to predict from.
        :param target: Optional target data to match to. If present, training is assumed.
        :return: Array (vector) of predicted results.
        """
        instance_index = 0
        full_set = []

        # Iterate through all instances.
        while instance_index != len(sample):
            col_index = 0
            indiv_set = self.bias

            # Iterate through all columns in instance. Should result with single value for instance.
            for column in sample[instance_index]:
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

    def train(self, training_data, num_steps):
        """
        Train the perceptron, performing num_steps weight updates.
        :param training_data: Data to train on. Includes both features and targets.
        :param num_steps: Number of iterations to update weights with.
        """
        # Determine number of items per training set.
        total_set_count = len(training_data)
        sets_per_step = total_set_count / num_steps
        logger.info('Number of Steps: {0}'.format(num_steps))
        logger.info('Total Set Count: {0}   Sets per Step: {1}'.format(total_set_count, sets_per_step))

        # Create result tracker class. Trains perceptron until no further improvement is found.
        tracker = ResultTracker()

        while tracker.continue_training_check():
            # Randomize so that every training iteration uses differently organized data.
            randomized_data = training_data.iloc[numpy.random.permutation(len(training_data))]
            train_x = randomized_data.loc[:, randomized_data.columns != 'medv'].values
            train_y = randomized_data['medv'].values

            # Iterate through sets and train perceptron accordingly.
            curr_index = 0
            high_index = 0
            self.total_errors = 0
            total_sets = 0
            while high_index < (total_set_count - 1):
                high_index += sets_per_step

                # Check that we don't exceed the max index.
                if high_index >= total_set_count:
                    high_index = (total_set_count - 1)

                # Train on the given sets.
                error_margin = 100
                set_iterations = 0
                sample_features = train_x[curr_index: math.floor(high_index)]
                sample_targets = train_y[curr_index: math.floor(high_index)]
                binary_targets = self._convert_to_binary_result(sample_targets)

                # Use current set until error margin is low or many iterations have occured.
                while error_margin > 30 and set_iterations < 1000:
                    results = self._train_step(sample_features, sample_targets)

                    binary_results = self._convert_to_binary_result(results)

                    # Calculate the error margin.
                    error_margin = self._determine_error_margin(binary_results, binary_targets, training=True)
                    total_sets += len(results)

                    # Keep track of number of times through given set.
                    # On occasion, randomized training data is bad and a "good" result is never found.
                    # This prevents locking up in such an instance.
                    set_iterations += 1

                curr_index = math.ceil(high_index)

            total_error_margin = ((self.total_errors / total_sets) * 100)
            logger.info('Total Errors: {0}   Total Sets {1}   Error Margin: {2}'
                        .format(self.total_errors, total_sets, total_error_margin))
            tracker.add_iteration(self.weights, total_error_margin)

        # In case the last iterations had worse results, grab best weights held by tracker.
        self.weights = tracker.iterations[tracker.best_iteration_index][0]
        best_error_margin = tracker.iterations[tracker.best_iteration_index][1]
        logger.info('')
        logger.info('Training Complete.')
        logger.info('Total Iterations: {0}'.format(len(tracker.iterations) - 1))
        logger.info(
            'Best Error Margin on Training Data was {0}% at iteration {1}.'
                .format(best_error_margin, tracker.best_iteration_index))
        logger.info('That means an accuracy of {0}%'.format(100 - best_error_margin))
        logger.info('Weights for said margin are {0}.'.format(self.weights))
        logger.info('')

    def _train_step(self, x, y):
        """
        Perform one training step:
            - Predict.
            - Calculate delta.
            - Update weights.
        :return: The predictions y_hat.
        """
        y_hat = self.predict(x)

        # Calculate binary of values and compare against targets.
        binary_results = self._convert_to_binary_result(y_hat)
        binary_targets = self._convert_to_binary_result(y)

        delta = self._delta(x, binary_results, binary_targets)
        self._update_weights(delta)
        return y_hat

    def _delta(self, features, results, target):
        """
        Given prediction results (y_hat) and targets (y), calculate the weight update delta.
        :param features: Initial features used in prediction.
        :param results: Prediction results, converted to binary.
        :param target: Expected target, converted to binary.
        :return: Full delta to update weights with.
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
                delta[weight_index] += ((target[row_index] - results[row_index]) * set[weight_index])
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
        :param delta: Weight update array (vector).
        """
        index = 0
        while index < len(delta):
            self.weights[index] += delta[index]
            index += 1

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

    def _determine_error_margin(self, results, targets, training=False, total=False):
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
        if not training:
            logger.info('Predicting...')
            logger.info('{0} predictions incorrect out of {1}. Error margin of {2}%'
                    .format(error_count, len(results), error_margin))
            logger.info('That means an accuracy of {0}%'.format(100 - error_margin))
            logger.info('Weights for said margin are {0}.'.format(self.weights))
            logger.info('')

        self.total_errors += error_count
        return error_margin


class ResultTracker():
    """
    Result tracker class. Tracks progress of perceptron.
    """
    def __init__(self):
        self.iterations = []
        self.best_iteration_index = 0

    def add_iteration(self, weights, error_margin):
        """
        Adds a new set of results to track.
        :param weights: Weights of current iteration.
        :param error_margin: Error margin of current iteration.
        """
        new_iteration = [weights, error_margin]
        self.iterations.append(new_iteration)
        logger.info('Iteration {0}: {1}'.format(len(self.iterations) - 1, new_iteration))

        logger.info('Previous Best: {0}   New Value: {1}'
                    .format(self.iterations[self.best_iteration_index][1], error_margin))

        # Calculate best iteration thus far. Based on total error margin.
        if error_margin < self.iterations[self.best_iteration_index][1]:
            self.best_iteration_index = len(self.iterations) - 1

    def continue_training_check(self):
        """
        Determines if perceptron should continue training.
        :return: True on continued training. False on training complete.
        """
        total_iterations = len(self.iterations)

        # Make perceptron iterate at least 10 times.
        if total_iterations <= 10:
            return True

        # Check if perceptron is still improving. Continue if progress has made in last 5 iterations.
        if self.best_iteration_index > (total_iterations - 5):
            return True

        return False
