from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from logger import Logger


def pca_cumulative_explained_variance(data, logger):

  logger.log("Normalize data using StandardScaler")
  std_scaler = StandardScaler()

  std_data = std_scaler.fit_transform(data)

  logger.log("Starting PCA on {} features...".format(data.shape[1]))
  pca = PCA(n_components=25).fit(std_data)
  logger.log("PCA done", show_time = True)

  sns.set()
  plt.figure(figsize=(5,4))
  plt.plot(np.cumsum(pca.explained_variance_ratio_))
  plt.xticks(range(0, 50, 5))
  plt.title("PCA cumulative explained variance depending of number of components")
  plt.xlabel("Number of components")
  plt.ylabel("Cumulative explained variance")
  plt.savefig(logger.get_output_file("pca_num_comp.png"), dpi = 120, 
    bbox_inches='tight')

  return pca


if __name__ == "__main__":

  logger = Logger(show = True, html_output = True, config_file = "config.txt")
  dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%y %H')
  train_filename = logger.config_dict['TRAIN_FILE']
  logger.log("Start reading training file {}...".format(train_filename))
  df_train = pd.read_csv(logger.get_data_file(train_filename), skipinitialspace = True, 
                         parse_dates = ['DATETIME'], date_parser = dateparse, index_col = 'DATETIME')
  df_train.sort_index(inplace = True)
  logger.log("Finish reading training file", show_time = True)

  data = df_train.loc[:, df_train.columns != 'ATT_FLAG'].values
  pca = pca_cumulative_explained_variance(data, logger)