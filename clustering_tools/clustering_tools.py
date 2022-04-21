"""This script contains clustering tools useful for the first assignment in the course"""

# TODO: a nem haszbnált importokat vedd ki
#  Ezt pl a PyCharm megteszi neked automatikusan, ha használod az Optimize Importsot
# TODO: az ipynb checkpoints mappa ne legyen része a repónak!
#  Ezt így csináld meg: https://stackoverflow.com/questions/343646/ignoring-directories-in-git-repositories-on-windows
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import cluster, datasets
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn import mixture
import seaborn as sns


def load_ski_resorts_data(data_path: str):
    """Load the ski resorts data, drop unnecessary columns and set the index"""
    return (
        pd.read_csv(data_path)
            .drop(columns=["Unnamed: 0", "Country", "Snowparks", "NightSki"])
            .set_index("Resort")
    )


def create_boxplot_for_feature_group(feature_group, group_name, inputData):
    # TODO: docstring
    fig, axs = plt.subplots(1, len(feature_group), figsize=(len(feature_group) * 5, 5))

    inputData[feature_group + ["CLUSTERING_1"]].boxplot(by="CLUSTERING_1", ax=axs)
    fig.suptitle(group_name.lower().replace("_", " "), y=1.05)
    plt.tight_layout()


def prepare_input_data_for_clustering(input_data, pca_components):
    """
    Steps to create a vlaid dataframe:
    * logarithmize the skewed vars
    * standardize the variables
    * Perform PCA
    * return the decomposed dataframe
    """
    input_data = remove_skew(input_data)
    input_data = standardize_columns(input_data)
    pca, input_data = perform_PCA(input_data, pca_components)
    return input_data, pca


def remove_skew(input_data):
    """Return the logarithm of skeqwed columns. This is usefuil for PCA"""
    for col in list(input_data.skew()[input_data.skew() > 1].index):
        input_data[col] = np.log(input_data[col] + 1)
    return input_data


def standardize_columns(input_data):
    """Standardize the columns"""

    return (input_data - input_data.mean()) / input_data.std()


def perform_PCA(input_data, n_components):
    """Return a decorrelated dataframe"""
    pca = PCA(n_components=n_components)
    input_data_pca = pca.fit_transform(input_data.values)
    return input_data_pca, pca
