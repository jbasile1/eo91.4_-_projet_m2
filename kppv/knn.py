#!/usr/bin/env python
# coding: utf-8

#==============================================================================
# Header (Duh)
#==============================================================================
# Jamasa REMY
# 25 November 2016
# K-Nearest Neighbors

#==============================================================================
# Credits
#==============================================================================
# This code was taken from
# http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

#===============================================================================
# What is k-Nearest Neighbors
#===============================================================================
# The model for kNN is the entire training dataset.
# When a prediction is required for a unseen data instance, the kNN algorithm
# will search through the training dataset for the k-most similar instances.
# The prediction attribute of the most similar instances is summarized and
# returned as the prediction for the unseen instance.
#
# The similarity measure is dependent on the type of data. For real-valued data,
# the Euclidean distance can be used. Other other types of data such as
# categorical or binary data, Hamming distance can be used.
#
# In the case of regression problems, the average of the predicted attribute may
# be returned.
# In the case of classification, the most prevalent class may be returned.

#===============================================================================
# How does k-Nearest Neighbors Work
#===============================================================================
# The kNN algorithm is belongs to the family of instance-based, competitive
# learning and lazy learning algorithms.
#
# Instance-based algorithms are those algorithms that model the problem using
# data instances (or rows) in order to make predictive decisions.
# The kNN algorithm is an extreme form of instance-based methods because all
# training observations are retained as part of the model.
#
# It is a competitive learning algorithm, because it internally uses competition
# between model elements (data instances) in order to make a predictive decision.
# The objective similarity measure between data instances causes each data
# instance to compete to “win” or be most similar to a given unseen data instance
# and contribute to a prediction.
#
# Lazy learning refers to the fact that the algorithm does not build a model until
# the time that a prediction is required. It is lazy because it only does work at
# the last second. This has the benefit of only including data relevant to the
# unseen data, called a localized model. A disadvantage is that it can be
# computationally expensive to repeat the same or similar searches over larger
# training datasets.
#
# Finally, kNN is powerful because it does not assume anything about the data,
# other than a distance measure can be calculated consistently between any two
# instances.
# As such, it is called non-parametric or non-linear as it does not assume a
# functional form.

import csv                      # For csv manipulation
import random                   # For generating pseudo-random numbers
import math                     # More maths functions
import operator                 # For transforming intrinsic opretators into
                                # function; operator.itemgetter
import sys                      # To read the command line arguments
import os
import matplotlib.pyplot as plt
import datetime
import getopt

from save import save

def load_datasets():
    """
    Loading the 2 data sets that were previously split.
    With [TODO : add the calculation that has been done on the sets]

    Params :
        training_set : the training set
        testing_set     : the testing set
    """
    testing_set  = [] # Will hold the test set
    training_set = [] # Will hold the training set

    # Loading the data from the test set
    with open('./data/testset.csv', 'rb') as csvfile:
        spamreader  = csv.reader(csvfile, delimiter = ',', quotechar = '|')
        testing_set = [row for row in spamreader]

    # Loading the data from the training set
    with open('./data/trainset.csv', 'rb') as csvfile:
        spamreader   = csv.reader(csvfile, delimiter = ',', quotechar = '|')
        training_set = [row for row in spamreader]

    # Deleting the columns names, they are not useful
    del(training_set[0])
    del(testing_set[0])

    # Converting the data set type from String to Float
    testing_set  = [map(float, row) for row in testing_set]
    training_set = [map(float, row) for row in training_set]

    return training_set, testing_set

def euclidean_distance(instance1, instance2, length):
    """
    Calculate the similarity between two data instances in order to make
    predictions.
    To locate the K most similar data instances in the training data set we use
    the Euclidean distance measure.

    The Euclidean distance is defined as the square root of the sum of the
    squared differences between the two arrays of numbers.

    One approach is to limit the euclidean distance to a fixed length, ignoring
    the final dimension.

    Params :
        instance1 : the first instance included in the distance calculation
        instance2 : the seconde instance included in the distance calculation
        length    : controls which fields to include in the distance calculation

    Output :
        the calculated distance between the instances
    """

    distance = 0 # Assuring that the distance is initialy 0

    for x in range(length): # We only want to include the first 4 attributes
        # Summing the squared difference of each field of the instances
        distance += (instance1[x] - instance2[x])

    return abs(distance)

def manhattan_distance(instance1, instance2, length):
    """
    Calculate the similarity between two data instances in order to make
    predictions.
    To locate the K most similar data instances in the training data set we use
    the Manhattan distance measure.

    The Manhattan distance is defined as the absolute value of the sum of the
    squared differences between the two arrays of numbers.

    One approach is to limit the euclidean distance to a fixed length, ignoring
    the final dimension.

    Params :
        instance1 : the first instance included in the distance calculation
        instance2 : the seconde instance included in the distance calculation
        length    : controls which fields to include in the distance calculation

    Output :
        the calculated distance between the instances
    """

    distance = 0 # Assuring that the distance is initialy 0

    for x in range(length): # We only want to include the first 4 attributes
        # Summing the squared difference of each field of the instances
        distance += pow((instance1[x] - instance2[x]), 2)

    return math.sqrt(distance)

def get_k_surrounding_neighbors(training_set, test_instance, k, distance_type =
                                'euclidean'):
    """
    This is a straight forward process of calculating the distance for all
    instances and selecting a subset with the smallest distance values.

    Params :
        training_set : the training set
        test_instance : the test set
        k            : the number of neighbors to retrieve

    Output :
        a list containing the k nearest neighbors to test_instance
    """

    distances = []  # A list that will contain the instances and thier distances
    length    = len(test_instance) - 1 # The number of fields in an instance

    for x in range(len(training_set)): # For each instance in the training_set
        # calculate the distance between a given test_instance and each element
        # in the training_set.
        #dist      = euclidean_distance(test_instance, training_set[x], length)
        if distance_type == 'euclidean':
            dist      = euclidean_distance(test_instance, training_set[x], length)
        elif distance_type == 'manhattan':
            dist      = manhattan_distance(test_instance, training_set[x], length)

        # Save the current instance and its distance from the given test_instance
        distances.append((training_set[x], dist))

    # Sorting the previously calculated distances from closest to furthest
    distances.sort(key = operator.itemgetter(1))
    # https://docs.python.org/2/library/operator.html
    # https://wiki.python.org/moin/HowTo/Sorting
    # http://pythoncentral.io/how-to-sort-a-list-tuple-or-object-with-sorted-in-python/

    neighbors = []
    # Retrieving the k nearest neighbors of test_instance
    for x in range(k):
        neighbors.append(distances[x][0])

    return neighbors

def make_prediction(neighbors):
    """
    Devise a predicted response based on those neighbors.
    Each neighbor will vote for their class attribute, and the majority
    vote will be use as the prediction.

    Below provides a function for getting the majority voted response from a
    number of neighbors.
    It assumes the class is the last attribute for each neighbor.

    Params :
        neighbors : the k neighbors for the requestested instance.

    Output :
        The class predicted class that we believ the requestested instance
        belongs to.
    """
    # Will contain the classes and the number of times they apprears amoung the
    # neighbors.
    class_votes = {}

    for x in range(len(neighbors)): # For each neighbors get
        response = neighbors[x][-1] # get its class.

        # Voting for a class
        if response in class_votes:    # If the class has already been taken
            class_votes[response] += 1 # into account then add another vote.
        else:                         # Otherwise
            class_votes[response]  = 1  # add the first vote for the actual class.

    # Sort the classes according to their amount of votes; from highest to the
    # one with the least vote.
    sorted_votes = sorted(class_votes.iteritems(), key = operator.itemgetter(1),
                         reverse = True)
    # Python2.7 : dict items vs dict iteritems
    # Originally, Python items() built a real list of tuples and returned that.
    # That could potentially take a lot of extra memory.

    # Then, generators were introduced to the language in general, and that
    # method was reimplemented as an iterator-generator method named iteritems().
    # The original remains for backwards compatibility.

    # Return the predicted class.
    return sorted_votes[0][0]

def get_accuracy(testing_set, predictions):
    """ THE CLASSIFICATION ACCURACY

    Evaluate the accuracy of the predictions by calculating the ratio of the total
    correct predictions out of all predictions made.

    Params :
        testing_set      : the set use for testing
        predictions  : the predictions made for each instances

    Output :
        The accuracy of the classification.
    """

    correct = 0

    for x in range(len(testing_set)): # For each instance check its classes equal to
        if testing_set[x][-1] == predictions[x]: # the one predicted.
            # If it does add one to the amount of correct answers.
            correct += 1

    # Return a percentage that reflects the accuracy of the classification.
    return (correct / float(len(testing_set))) * 100.0

def get_accuracy_by_class(testing_set, predictions):
    """ THE CLASSIFICATION ACCURACY FOR EACH CLASS

    Evaluate the accuracy of the predictions by calculating the ratio of the total
    correct predictions out of all predictions made for each class.

    Params :
        testing_set  : the set use for testing
        predictions  : the predictions made for each instances

    Output :
        the amount of unique classes found in the set
        the total of classes found in the set (per instance)
        the success rate for each class
    """

    training_set_class = list(set([i[-1] for i in testing_set]))
    instances_class    = [float(i[-1]) for i in testing_set]
    instance_per_class = {i: instances_class.count(i) for i in instances_class}
    prediction_per_class = {}

    # Initializing the number of correct predictions per classes to 0
    for cls in training_set_class:
        prediction_per_class[int(cls)] = 0

    for x in range(len(testing_set)): # For each elements in the set
        # Check if the predictions is correct
        if instances_class[x] == predictions[x]:
            # If the classe has already been seen then add one
            prediction_per_class[instances_class[x]] += 1

    # The number of intances per class
    percentage_per_class = dict(prediction_per_class)

    # Calculat the success rate for each class
    for key in percentage_per_class.keys():
        # Count the number of times the appears in the set
        nb_instances_for_class = instances_class.count(key)

        # Calculate the percentage of success
        percentage_per_class[key] = ((percentage_per_class[key] /
                                   float(nb_instances_for_class)) * 100.0)

    return (len(training_set_class), len(instances_class), prediction_per_class,
            percentage_per_class, instance_per_class)

def output(predictions, testing_set):
    """
    """

    print '\nPredictions predictionss :'
    for x in range(len(testing_set)):
        if predictions[x] == testing_set[x][-1]:
            print('\tpredicted : ' + repr(predictions[x]) + ' <=> actual : ' +
                  repr(testing_set[x][-1]))
        else:
            print('\t' + '\x1b[0;0;31m' + 'predicted : ' + repr(predictions[x]) +
                  ' <=> actual : ' + repr(testing_set[x][-1]) + '\x1b[0m')

def main():
    # Start : Parsing the arguments
    # The list of accepted arguments
    short_args = 'k:d:v:s:h'
    long_args  = ['neighbors=', 'distance=', 'verbose=', 'save=', 'help=']

    # The classification cutomization variables
    verbose       = False
    save          = False
    k             = 3
    distance_type = 'manhattan'

    try:
        options, remainder =  getopt.getopt(sys.argv[1:], short_args, long_args)

        for opt, arg in options:
            if opt in ('-k', '--neighbors'):
                k = int(arg)
            elif opt in ('-d', '--distance'):
                distance_type = str(arg)
            elif opt in ('-v', '--verbose'):
                verbose = bool(arg)
            elif opt in ('-s', '--save'):
                save = bool(arg)
            elif opt in ('-h', '--help') :
                print '-k or --neighbors\n\tThe number of neighbors to take account of.'
                print '-d or --distance\n\tThe distance to use.\n\t\tmanhattan\n\t\teuclidean'
                print '-v or --verbose\n\tPrint the predictions to the screen.'
                print '-s or --save\n\tSave the results to the harddrive.'
                print '-h or --help\n\tPrint this help menu.'
                sys.exit(1)
    except getopt.GetoptError, msg:
        print 'Error :', msg
        print '-h or --help\n\tTo list the list of options.'
        sys.exit(2)
    # End

    # Prepare the data
    training_set, testing_set  = load_datasets() # The split sets
    predictions = []                             # Generate predictions


    # Printing some useful information
    print 'Train sets size  :', repr(len(training_set))
    print 'Test sets size   :', repr(len(testing_set))
    print 'K                :', k
    print 'Type of distance :', distance_type

    # Get the predictions
    for x in range(len(testing_set)):
        neighbors = get_k_surrounding_neighbors(training_set, testing_set[x], k,
                                                distance_type)
        result    = make_prediction(neighbors)

        predictions.append(result)

    # Calculate and print the accuracy of the classification
    accuracy = get_accuracy(testing_set, predictions)
    print('\nFull accuracy        : ' + repr(accuracy) + '%')

    # Print the accuracy for each class
    accuracy_by_class = get_accuracy_by_class(testing_set, predictions)
    print '\nClasses found in the set :', accuracy_by_class[0]
    print 'Size of the set          :', accuracy_by_class[1]
    print 'Accuracy per class'

    for pc in accuracy_by_class[2].keys():
        print '\tClass', pc, '(' + repr(accuracy_by_class[2][pc]) + '/' + \
                                   repr(accuracy_by_class[4][pc]) + ')', ':', \
                                   repr(accuracy_by_class[3][pc]) + '%'

    # If print the results if the 3rd argument is 'o'
    if verbose == True:
        output(predictions, testing_set)
    elif save == True:
        save(predictions, testing_set, training_set, k, accuracy, accuracy_by_class)

# Run the classification
if __name__ == '__main__':
    main()
