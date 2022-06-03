import numpy as np
import sklearn.ensemble
import matplotlib.pyplot as plt
import pandas as pd
from pointers import *
import pdb

def load_results(filename = location_hardness_models, sep = '\t', header = 40):
    '''
    This function loads sources from our CSC results tsv and outputs a Pandas data 
    structure to be used in the random forest training.
    '''
    Cat = pd.read_csv(filename, sep = sep, header = header)
    # usable_sources = Cat[(np.abs(Cat['hard_hs']) < 0.995) & (np.abs(Cat['hard_hm']) < 0.995) & \
    #     (np.abs(Cat['hard_ms'])<0.995)]
    usable_sources = Cat
    return usable_sources


def sig_source_subset(catalog, min_sig, max_sig):
    selection_index = np.logical_and(catalog['significance'] >= min_sig, \
        catalog['significance'] < max_sig)
    return catalog[selection_index].copy()


def source_subset(usable_sources, N = 6e3, seed = None):
    source_total = float(len(usable_sources['hard_hs']))    
    crit_val = N/source_total
    np.random.seed(seed)
    selection_val = np.random.random(int(source_total))
    selection_index = selection_val < crit_val
    return usable_sources[selection_index].copy()

def feature_list():
    feature_list = ['gal_l','gal_b','err_ellipse_r0','err_ellipse_r1','err_ellipse_ang',\
    'significance','hard_hm','hard_hs','nh_gal','acis_num','acis_time','likelihood',\
    'flux_powlaw','powlaw_gamma','powlaw_nh','powlaw_ampl','powlaw_stat',\
    'flux_bb','bb_kt','bb_nh','bb_ampl','bb_stat','flux_brems','brems_kt','brems_nh',\
    'brems_norm','brems_stat']
    return feature_list


def extract_values(sources, features = feature_list()): 
    training_features = sources.loc[:,features].values
    
    # The random forest tools in scikit-learn are in 32 bit, so we need to make
    # sure there are no numbers that can't be represented in 32 bits.
    np.clip(training_features, -9999, 2e6, out = training_features)
    return training_features

def save_significance_bins():
    catalog = load_results()
    sig = np.copy(catalog['significance'])
    bin_walls = np.array([3, 4.5, 6, 7.5, max(sig) + 1])
    
    for i in np.arange(0,len(bin_walls)-1):
        binmin = bin_walls[i]; binmax = bin_walls[i+1]
        filename = 'significance_bin__' + str(binmin) + '--' + str(binmax)
        sig_bin = catalog[np.logical_and(sig >= binmin, sig < binmax)]
        sig_bin.to_csv(path_or_buf = results_path + filename + '.csv')

def clean_data(data):
    '''
    Function that cleans the input Pandas data frame. The process looks for features
    that have missing values (represented by -9999), creates a new feature named 
    missing_[feature name] consisting of 0 (False) or 1 (True), then replaces the 
    missing values (-9999) with a 0.

    Input:
        data: A Pandas dataframe with missing values represented by -9999

    Output:
        cleaned_data: A copy of the input data that has been cleaned using the 
                      described process
        
        cleaned_feature_list: The updated feature list including the new features
    '''

    # Checking to make sure the input data hasn't already been cleaned. Without this
    # step, cleaning cleaned data would erase the missing_feature-name features
    current_features = data.columns.to_numpy()
    if any('missing' in header for header in current_features):
        return data, current_features

    cleaned_data = data.loc[:, feature_list()].copy()
    cleaned_feature_list = feature_list()
    headers = cleaned_data.columns.to_numpy()
    
    model_list = ['flux_powlaw','powlaw_gamma','powlaw_nh','powlaw_ampl','powlaw_stat',\
    'flux_bb','bb_kt','bb_nh','bb_ampl','bb_stat','flux_brems','brems_kt','brems_nh',\
    'brems_norm','brems_stat']
    model_checked = False

    for header in headers:
        if sum(cleaned_data[header] == -9999) > 0:
            if header in model_list:
                if model_checked is False:
                    missing_feature = 'missing_models'
                    model_checked = True
                else:
                    continue
            else:
                missing_feature = 'missing_' + header

            cleaned_data[missing_feature] = pd.Series([0 for x in range(len(cleaned_data.index))], \
                index=cleaned_data.index)
            cleaned_data.at[cleaned_data[header] == -9999, missing_feature] = 1
            cleaned_feature_list.append(missing_feature)

    cleaned_data[cleaned_data == -9999] = 0
    return cleaned_data, cleaned_feature_list