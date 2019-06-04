from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd

from pca_outliers import pca_get_outliers
from logger import Logger


def pca_get_best_threshold(train_data, optim_data, optim_labels, n_components, thresholds_to_test, 
  logger):

  logger.log("Normalize data using StandardScaler")
  std_scaler = StandardScaler()

  train_std_data = std_scaler.fit_transform(train_data)
  optim_std_data = std_scaler.transform(optim_data)

  logger.log("Starting PCA on {} features...".format(train_data.shape[1]))
  pca = PCA(n_components = n_components).fit(train_std_data)
  logger.log("PCA with {} components done".format(n_components), show_time = True)

  # first 12 for modeling normal subspace
  P = pca.components_[:12].T

  # formulas as from network traffic anomalies paper
  C = np.dot(P, P.T)
  y_residual = [np.dot(np.identity(43) - C, y_elem.T) for y_elem in optim_std_data]

  optim_rate = {}
  for threshold in thresholds_to_test:
    tp, fp, fn = 0, 0, 0
    for idx, crt_obs in enumerate(optim_std_data):
      crt_error = sum(y_residual[idx] - crt_obs) ** 2

      if crt_error > threshold and int(optim_labels[idx]) != 1:
        fp += 1
      elif crt_error > threshold and int(optim_labels[idx]) == 1:
        tp += 1

    ratio = tp / fp if fp != 0 else 1
    # enforcing solutions with more TP
    if tp < 10:
      ratio = 0

    optim_rate[threshold] = ratio

    logger.log("Threshold {} with {}/{} TP/FP".format(threshold,  tp, fp))

  print(optim_rate)
  best_threshold = sorted(optim_rate.items(), key=lambda tup: tup[1])[-1][0]
  logger.log("Best threshold is {}".format(best_threshold))


if __name__ == "__main__":

  logger = Logger(show = True, html_output = True, config_file = "config.txt")
  dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%y %H')
  train_filename = logger.config_dict['TRAIN_FILE']
  logger.log("Start reading training file {}...".format(train_filename))
  df_train = pd.read_csv(logger.get_data_file(train_filename), skipinitialspace = True, 
                         parse_dates = ['DATETIME'], date_parser = dateparse, index_col = 'DATETIME')
  df_train.sort_index(inplace = True)
  logger.log("Finish reading training file", show_time = True)

  optim_filename = logger.config_dict['OPTIM_FILE']
  logger.log("Start reading optimization file {}...".format(optim_filename))
  df_optim = pd.read_csv(logger.get_data_file(optim_filename), skipinitialspace = True, 
                         parse_dates = ['DATETIME'], date_parser = dateparse, index_col = 'DATETIME')
  df_optim.sort_index(inplace = True)
  logger.log("Finish reading optimization file", show_time = True)
  
  # as resulted from pca_components.py
  n_components = 15

  optim_data = df_optim.loc[:, df_optim.columns != 'ATT_FLAG']
  optim_labels = df_optim['ATT_FLAG']
 
  # eliminate outliers as seen in pca_outliers.py
  train_data = df_train.loc[:, df_train.columns != 'ATT_FLAG']
  outliers_idxs = pca_get_outliers(train_data, n_components, logger)
  df_train_indexes = df_train.index.values[outliers_idxs]
  df_train.drop(df_train_indexes, inplace = True)
  train_data = df_train.loc[:, df_train.columns != 'ATT_FLAG']


  thresholds_to_test = range(100, 420, 20)
  pca_get_best_threshold(train_data, optim_data, optim_labels, n_components, 
    thresholds_to_test, logger)