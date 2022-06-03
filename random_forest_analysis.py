import numpy as np
import sklearn.ensemble
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
from pointers import *
import random_forest_utilities as rfu
import pdb
style.use('seaborn-whitegrid')



def return_synthetic_data(data_real):
    """
    The function returns a matrix with the same dimensions as data_real but with synthetic data
    based on the marginal distributions of its featues
    """
    features = len(data_real[0])
    data_synth = np.zeros(data_real.shape)

    for i in range(features):
        obs_vec = data_real[:,i]
        # here we chose the synthetic data to match the marginal distribution of the real data
        syn_vec = np.random.choice(obs_vec, len(obs_vec)) 
        data_synth[:,i] += syn_vec

    return data_synth


def merge_real_and_synthetic_samples(data_real, data_synth):
    """
    The function merges the data into one sample, giving the label "1" to the real 
    data and label "2" to the synthetic data.
    """
    # build the labels vector
    labels_real = np.ones(len(data_real))
    labels_synth = np.ones(len(data_real))*2

    labels_total = np.concatenate((labels_real, labels_synth))
    data_total = np.concatenate((data_real, data_synth))
    return data_total, labels_total


def build_similarity_matrix(random_forest, data_real):
    """
    The function builds the similarity matrix based on the feature matrix data_real for the
    results labels_real based on the trained random forest.
    The matrix is normalised so that the biggest similarity is 1 and the lowest is 0.

    This function counts only leaves in which the object is classified as a "real" object 
    it is also implemented to optimize running time, asumming one has enough running memory.
    """
    # apply to get the leaf indices
    print('Applying random forest to data...')
    apply_mat = random_forest.apply(data_real)
    # find the predictions of the sample
    is_good_matrix = np.zeros(apply_mat.shape)
    print('Checking predictions...')
    for i, est in enumerate(random_forest.estimators_):
        d = est.predict_proba(data_real)[:, 0] == 1
        is_good_matrix[:, i] = d
    # mark leaves that make the wrong prediction as -1, in order to remove them 
    # from the distance measurement
    apply_mat[is_good_matrix == False] = -1 
    # now calculate the similarity matrix
    print('Calculating similartity matrix...')
    # pdb.set_trace()
    sim_mat = np.sum((apply_mat[:, None] == apply_mat[None, :]) & (apply_mat[:, None] != -1) \
        & (apply_mat[None, :] != -1), axis=2) / np.asfarray(np.sum([apply_mat != -1], axis=2), \
        dtype='float')
    return sim_mat


def train_random_forest(Sources, features = rfu.feature_list(), N_trees = 500, min_samples = 1, \
    criterion = 'gini'):
    
    training_features = rfu.extract_values(Sources, features)
    print('Features selected.')

    synthetic_matrix = return_synthetic_data(training_features)
    print('Synthetic matrix created.')

    training_matrix, categories = merge_real_and_synthetic_samples(training_features, \
        synthetic_matrix)
    print('Training matrix constructed.')

    print('Growing random forest...')
    random_forest = sklearn.ensemble.RandomForestClassifier(n_estimators = N_trees, \
        min_samples_leaf= min_samples, criterion = criterion)
    print('Random forest grown.')
    
    print('Training random forest...')
    random_forest.fit(training_matrix, categories)
    print('Random forest trained.')
    
    return training_features, random_forest


def find_distance_measurements(random_forest,sources):

    print('Building similarity matrix...')
    similarity_matrix = build_similarity_matrix(random_forest, sources)
    print('Similarity matrix completed...')
    # pdb.set_trace()
    distance_matrix = 1 - similarity_matrix
    sum_vec = np.sum(distance_matrix, axis=1)
    sum_vec /= float(len(sum_vec))
    
    return sum_vec

def plot_weirdness_histogram(sum_vec, features, plot_name, directory_path = None):

    f = plt.figure(figsize = (12,8))
    tmp = plt.hist(sum_vec, bins=60, color="g", edgecolor = 'k')
    plt.ylabel("N")
    plt.xlabel("weirdness score")
    feature_list = '--'.join(features)
    plt.title(plot_name)
    # f.text(.5, 0.5, feature_list, ha = 'center')
    if directory_path == None:
        filename = './weirdness_histograms/' + plot_name + '.pdf'
    else:
        filename = directory_path + plot_name + '.png' 
    f.savefig(filename, overwrite = True)
    
    print('Histogram saved as ' + filename)


def produce_weirdness_histogram(random_forest, sources, features, plot_name):
    
    sum_vec = find_distance_measurements(random_forest, sources)
    
    plot_weirdness_histogram(sum_vec, features, plot_name)


def find_outliers(random_forest, sources, weirdest_percent = 0.01):
    
    sum_vec = find_distance_measurements(random_forest, sources)
    threshold = max(sum_vec) - weirdest_percent*(max(sum_vec) - min(sum_vec))
    return sum_vec > threshold

def ball_pit(catalog, N = 6000, seed = None, N_trees = 500, min_samples = 1, criterion = 'gini'):
    cleaned_data, features = rfu.clean_data(catalog)
    distance_score = np.zeros(len(catalog['significance']))

    training_set = rfu.source_subset(cleaned_data, N = N, seed = seed)
    random_forest = train_random_forest(training_set, features = features, N_trees = N_trees, \
        min_samples = min_samples, criterion = criterion)[1]
    
    '''
    Here we need to find a way to randomly select subsets of the full catalog, 
    apply the RF to them, find the distances, and then move on to the next subset.
    We do this by assigning a random number to each source and then processing
    the batches with matching numbers. 
    '''
    n_baskets = int(round(len(cleaned_data['significance'])/float(N),0))
    queue = np.random.randint(1, n_baskets + 1, len(cleaned_data['significance']))
    for i in np.arange(1, n_baskets + 1):
        print("\nBatch " + str(i) + ' of ' + str(n_baskets) + '\n')
        index = queue == i
        print(str(sum(index)) + ' sources')
        batch = rfu.extract_values(cleaned_data[index], features = features)
        batch_distances = find_distance_measurements(random_forest, batch)
        distance_score[index] = batch_distances
    
    catalog['distance_score'] = distance_score
    return catalog

def plot_feature_importance(sources, features = rfu.feature_list(), iterations = 1, \
    save = True, filename = None, filepath = None, N_trees = 500, min_samples = 1, \
    criterion = 'gini'):
    '''
    Function for plotting the feature importance of a trained random forest as a 
    horizontal bar graph.

    Input:
        sources: a Pandas dataframe of the full catalog (source subsets will be 
                 made within the function)

        features: a list or Numpy array of the features used for training. 
                  default = rfu.feature_list()
        
        interations: integer for the number of random forests to be trained and 
                     tested. default = 1
        
        save: boolean indicating whether plot image should be saved. default = True
        
        filename: string indicating the name of the file to be saved, including 
                  the file extension. default = None

        filepath: string indicating the filepath where the image should be saved.
                  default = None. If left as default, image will be saved in the 
                  working directory.

    Output:
        A horizontal bar plot (optionally saved) ranking the features in descending 
        order of importance. Plot will include standard deviations (for multiple 
        iterations).
    '''
    print('Total iterations: ' + str(iterations) + '\n')
    print('Figure to be saved: ' + str(save) + '\n')
    print('Filename and extension: ' + str(filename) + '\n')

    importances = np.zeros((iterations, len(features)))
    for i in np.arange(iterations):
        print('feature importance iteration: ' + str(i + 1) + ' of ' + str(iterations) + '\n')
        subset = rfu.source_subset(sources, seed = i)
        rf = train_random_forest(subset, features = features, N_trees = N_trees, \
            min_samples = min_samples, criterion = criterion)[1]
        importances[i] = rf.feature_importances_
    mean = np.mean(importances, axis = 0); std = np.std(importances, axis = 0)
    rank = np.argsort(mean)

    f = plt.figure(figsize = (15,22))
    plt.barh(np.array(features)[rank], mean[rank], xerr = std[rank])
    
    if iterations == 1:
        plt.title('Feature Importances: 1 Iteration', size = 25)
    else:
        plt.title('Feature Importances: ' + str(iterations) + ' Iterations', size = 25)
    
    plt.ylabel('Feature', size = 20); plt.xlabel('Importance Score (%)', size = 20)
    plt.yticks(size = 15); plt.xticks(size = 15)
    
    if save is True:
        if filename is None:
            from datetime import datetime
            filename = 'feature_importance_iteration_' + str(iterations) + '_' \
                + datetime.now().strftime("%Y-%m-%d-%H_%M_%S") + '.png'
            if filepath is None:
                f.savefig('./' + filename)
            else:
                f.savefig(filepath + '/' + filename)
        else:
            if filepath is None:    
                f.savefig('./' + filename, overwrite = True)
            else: 
                f.savefig(filepath + '/' + filename)
        print("Plot saved as " + filename)
    
    # f.show()
    return