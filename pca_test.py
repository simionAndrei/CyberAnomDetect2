from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd

from pca_outliers import pca_get_outliers
from logger import Logger


def pca_get_attacks(train_data, test_data, test_labels, n_components, threshold, logger):

  logger.log("Normalize data using StandardScaler")
  std_scaler = StandardScaler()

  train_std_data = std_scaler.fit_transform(train_data)

  logger.log("Starting PCA on {} features...".format(train_data.shape[1]))
  pca = PCA(n_components = n_components).fit(train_std_data)
  logger.log("PCA with {} components done".format(n_components), show_time = True)

  P = pca.components_[:12].T
  C = np.dot(P, P.T)

  test_std_data = std_scaler.transform(test_data)

  y_residual = [np.dot(np.identity(43) - C, y_elem.T) for y_elem in test_std_data]

  tp, fp = 0, 0
  for idx, crt_obs in enumerate(test_std_data):
    crt_error = sum(y_residual[idx] - crt_obs) ** 2

    if crt_error > threshold and int(test_labels[idx]) != 1:
      fp += 1
    elif crt_error > threshold and int(test_labels[idx]) == 1:
      tp += 1

  logger.log("Results on test data: {}/{} TP/FP".format(tp, fp))

  

if __name__ == "__main__":

  logger = Logger(show = True, html_output = True, config_file = "config.txt")
  dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%y %H')
  train_filename = logger.config_dict['TRAIN_FILE']
  logger.log("Start reading training file {}...".format(train_filename))
  df_train = pd.read_csv(logger.get_data_file(train_filename), skipinitialspace = True, 
                         parse_dates = ['DATETIME'], date_parser = dateparse, index_col = 'DATETIME')
  df_train.sort_index(inplace = True)
  logger.log("Finish reading training file", show_time = True)

  test_filename = logger.config_dict['TEST_FILE']
  logger.log("Start reading test file {}...".format(test_filename))
  df_test = pd.read_csv(logger.get_data_file(test_filename), skipinitialspace = True, 
                         parse_dates = ['DATETIME'], date_parser = dateparse, index_col = 'DATETIME')
  df_test.sort_index(inplace = True)
  logger.log("Finish reading test file", show_time = True)
  
  # as resulted from pca_components.py
  n_components = 15

  # as resulted from pca_threshold
  threshold = 160

  test_data = df_test.loc[:, df_test.columns != 'ATT_FLAG']
  test_labels = df_test['ATT_FLAG']
 
  train_data = df_train.loc[:, df_train.columns != 'ATT_FLAG']
  outliers_idxs = pca_get_outliers(train_data, n_components, logger)
  df_train_indexes = df_train.index.values[outliers_idxs]
  df_train.drop(df_train_indexes, inplace = True)
  train_data = df_train.loc[:, df_train.columns != 'ATT_FLAG']

  pca_get_attacks(train_data, test_data, test_labels, n_components, threshold, logger)