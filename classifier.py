# Classifier.py
# Author: Jeffrey Lin
# Date: 4/10/2023
# 
#
# Purpose: This program utilizes two different methods of classification to 
# predict whether patients in the test sample are either current smokers or 
# have never smoked before. The methods of classification used in this program 
# include k-nearest-neighbors and agglomerative clustering.

import sys as sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import pandas as pd
import math as math 
from sklearn.cluster import AgglomerativeClustering

# accuracy
# purpose: Calculates the accuracy of the predicted labels compared to actual 
#          or true labels  
# input: array storing the true labels and an array storing the predicted
#        labels
# output: float representing the percent accuracy
# Effects: None

def accuracy(y_true, y_pred):
        match = 0

        y_pred = y_pred.tolist()
        y_true = y_true.tolist()

        for label in range(len(y_pred)):
            if y_pred[label] == y_true[label]:
                match += 1

        return match / len(y_pred)

# write_to_output
# purpose: function for writing output to a file
# arguments: list representing content of the file, name of file that is to 
#            to be written, dataframe containing indices of the data
# returns: none
# effects: writes content of some list to a file

def write_to_output(file_content, file_name, test_data):
    file = open(file_name, "w")
    file_content.tolist()

    indices = test_data.iloc[:,0].index.tolist()

    for label in range(len(file_content)):
        file.write(indices[label] + " " + file_content[label] + "\n")

# cross_validate_all_k
# purpose: tests the accuracy of the knn classifier by cross-validating with 
#          5 folds from the training data and using series of different k-values
# arguments: x_training data and y_training data
# returns: returns dictionary of predicted labels keyed by k-value
# effects: creates scatterplot of accuracies of classification by k-value

def cross_validate_all_k(x_train, y_train):
        k_vals = [1, 3, 5, 7, 11, 21, 23]
        Partition_list = [6, 12, 18, 24, 30]

        y_pred_by_k = dict()
        acc_by_k = dict()

        # determines predicted labels and accuracies of prediction for all 5 
        # fold for all k-values
        for k in range(len(k_vals)):
            fold_acc_list = []
            pred_labels = []

            for partition in range(len(Partition_list)): 
                start = Partition_list[partition] - 6
                stop = start + 6

                # witholds portion of data according to fold
                indices = [i for i in range(start,stop)]
                training_x = np.delete(x_train, indices, axis = 0)
                training_y = np.delete(y_train, indices, axis = 0)
                
                # performs cross validation for given fold
                knn = KNNClassifier(k_vals[k], training_x, training_y)
                pred_label, fold_accuracy = \
                    knn.cross_validate(y_train[start:stop], x_train[start:stop])

                # compiles predicted labels from each fold
                pred_labels = pred_labels + pred_label
                fold_acc_list.append(fold_accuracy)

            
            classifier_accuracy = sum(fold_acc_list) / len(fold_acc_list)
            
            y_pred_by_k[k_vals[k]] = pred_labels
            acc_by_k[k_vals[k]] = classifier_accuracy

        # create list of accuracies to use for plotting
        accuracies = []
        for val in k_vals:
            accuracies.append(acc_by_k[val])

        
        # plots the accuracies of predicted labels by the k-value
        title = "Accuracy of Classification according to K neighbors"
        plt.scatter(k_vals, accuracies , s = 8)
        plt.title(title, fontsize = 10)
        plt.xlabel("Number of Neighbors", fontsize = 10)
        plt.ylabel("Accuracy of Classification", fontsize = 10)
        plt.savefig("knn_accuracies.png")

        return y_pred_by_k

# calc_TP_TN_FP_FN
# purpose: Determines number of true positive, true negatives, false positives, 
#          and false negatives from some classifier
# arguments: array containing true labels and array containing predicted labels
# returns: dictionary containing counts for each of the 4 categories
# effects: creates dictionary storing TP, TN, FP, FN

def calc_TP_TN_FP_FN(y_true, y_pred):
    TP_TN_FP_FN_dict = {
        "TP": 0,
        "TN": 0,
        "FN": 0,
        "FP": 0
    }

    for i in range(len(y_true)):
        if (y_true[i] == "CurrentSmoker" and y_pred[i] == y_true[i]):
            TP_TN_FP_FN_dict["TP"] += 1
        elif (y_true[i] == "NeverSmoker" and y_pred[i] == y_true[i]):
            TP_TN_FP_FN_dict["TN"] += 1
        elif (y_true[i] == "CurrentSmoker" and y_pred[i] != y_true[i]):
            TP_TN_FP_FN_dict["FN"] += 1
        elif (y_true[i] == "NeverSmoker" and y_pred[i] != y_true[i]):
            TP_TN_FP_FN_dict["FP"] += 1

    return TP_TN_FP_FN_dict

# print_TP_TN_FP_FN
# purpose: print content of TP_TN_FP_FN dictionary
# arguments: TP_TN_FP_FN dictionary
# returns: none
# effects: prints out the counts of TP, TN, FP, FN

def print_TP_TN_FP_FN(TP_TN_FP_FN_dict):
    Class = ["TP", "TN", "FP", "FN"]

    for label in Class:
        print(label + ": " + str(TP_TN_FP_FN_dict[label]))

class KNNClassifier:

    # constructor
    # purpose: initializes instance of KNN classifier
    # arguments: k-value, x_training data, y_training data
    # returns: none
    # effects: creates instance of KNN classifier

    def __init__(self, k,  x_train, y_train):
        self.k = k
        self.x_train = x_train 
        self.y_train = y_train

    # predict
    # purpose: predicts labels of test data
    # arguments: test data
    # returns: list of labels for test data
    # effects: none

    def predict(self, x_test):
        pred_y = np.empty([len(x_test)], dtype = '<U25')

        test = x_test
        
        for patient in range(len(test)):
            Class_label = self.get_euclidean_dist(test, self.x_train, patient)
            pred_y[patient] = Class_label

        return pred_y

    # get_euclidean_dist
    # purpose: determines euclidean distance between test data and training data
    # arguments: test data, training data, patient from test data
    # returns: predicted label for given patient
    # effects: none
    def get_euclidean_dist(self, test_sample, train_sample, patient):
        dist_list = []

        # iterates through all patients in training data
        for j in range(len(train_sample)):
            
            # calculates euclidean distance
            sample_dist = np.linalg.norm(test_sample[patient] - train_sample[j])
            dist_list.append((sample_dist, self.y_train[j]))

        # sorts the euclidean distances to find the smallest distances
        df = pd.DataFrame(dist_list, columns = ["Euclidean distance", "Class"])
        df = df.sort_values(by=['Euclidean distance'])

        return self.get_label(df)
        
    # get_label
    # purpose: finds majority-rule label according to euclid's distance list
    # arguments: list of sorted euclidean distances
    # returns: predicted label
    # effects: none

    def get_label(self, sorted_euclids):
        neighbors = sorted_euclids.iloc[:int(self.k), 1]
        
        max_class = neighbors.value_counts()

        return max_class.index.to_list()[0]
    
    # cross_validate
    # purpose: predicts labels for test data and determines its accuracy
    # arguments: true labels, test data
    # returns: list of predicted labels, the accuracy of said predictions
    # effects: none
    
    def cross_validate(self, y_true, test_data):
        y_pred = self.predict(test_data)
        percent_acc = accuracy(y_true, y_pred)

        return y_pred.tolist(), percent_acc


def main():
    # Read in training and test data
    k_neighbors = sys.argv[1]
    training_data = pd.read_csv('GSE994-train.txt', sep = "\t")
    test_data = pd.read_csv('GSE994-test.txt', sep = "\t")

    # Formatting data
    training_data = training_data.transpose()
    test_data = test_data.transpose()

    test_data_indices = test_data
    
    test_data = test_data.to_numpy()
    y_train = training_data.loc[:, "Class"]
    y_train = y_train.to_numpy()
    x_train = training_data 
    x_train = x_train.to_numpy()

    length = len(test_data[0])
    x_train = x_train[:, :length - 1]
    test_data = test_data[:, :length -1]

    test_data = test_data.astype(float)
    x_train = x_train.astype(float)
    
    # predicts labels for test data
    knn = KNNClassifier(k_neighbors, x_train, y_train)
    y_pred = knn.predict(test_data)
    file_name = "Prob5-" + k_neighbors + "NNoutput.txt"
    write_to_output(y_pred, file_name, test_data_indices)

    # tests accuracy of KNN through cross-validation
    y_pred_by_k_knn = cross_validate_all_k(x_train, y_train)
        
    # predicts labels on training data through Agglomerative Clustering
    y_pred_by_k_cluster = AgglomerativeClustering(n_clusters = 2, 
                        linkage = 'average').fit_predict(x_train)
    y_pred_by_k_cluster = \
        np.where(y_pred_by_k_cluster == 1, "CurrentSmoker", "NeverSmoker")
   
    cluster_accuracy = accuracy(y_train, y_pred_by_k_cluster)

    # Determines num of TP, TN, FP, and FN for each classification method
    cluster_pos_neg = calc_TP_TN_FP_FN(y_train, y_pred_by_k_cluster)
    KNN_pos_neg = calc_TP_TN_FP_FN(y_train, y_pred_by_k_knn[5])

    print("KNN Method")
    print_TP_TN_FP_FN(KNN_pos_neg)
    print("Agglomerative Clustering Method")
    print_TP_TN_FP_FN(cluster_pos_neg)

if __name__ == "__main__":
    main()
