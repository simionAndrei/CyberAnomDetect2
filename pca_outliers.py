from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from logger import Logger

def plot_residual(datetime_index, residual, filename, logger):

  sns.set()
  plt.figure(figsize=(15,10))

  plt.plot(datetime_index, residual)
  plt.yticks(range(0, 10000, 500))
  plt.title("Residual on training data")
  plt.xlabel("Time")
  plt.ylabel("Residual")

  plt.savefig(logger.get_output_file(filename), dpi = 120, 
    bbox_inches='tight')


def pca_get_outliers(data, n_components, logger):

  logger.log("Normalize data using StandardScaler")
  std_scaler = StandardScaler()

  std_data = std_scaler.fit_transform(data)

  logger.log("Starting PCA on {} features...".format(data.shape[1]))
  pca = PCA(n_components = n_components).fit(std_data)
  logger.log("PCA with {} components done".format(n_components), show_time = True)

  P = pca.components_[:12].T
  C = np.dot(P, P.T)

  y_residual = [np.dot(np.identity(43) - C, y_elem.T) for y_elem in std_data]

  # determined by visualizing residual plot on training data
  threshold = 500
  squared_pred_errors, preds = [], []
  for idx, crt_obs in enumerate(std_data):
    crt_error = sum(y_residual[idx] - crt_obs) ** 2

    if crt_error > threshold:
      preds.append(1)
    else:
      preds.append(0)

    squared_pred_errors.append(crt_error)

  plot_residual(data.index, squared_pred_errors, "outliers_residuals.png", logger)

  outliers_idxs = np.where(np.array(preds) == 1)[0].tolist()

  return outliers_idxs


if __name__ == "__main__":

  logger = Logger(show = True, html_output = True, config_file = "config.txt")
  dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%y %H')
  train_filename = logger.config_dict['TRAIN_FILE']
  logger.log("Start reading training file {}...".format(train_filename))
  df_train = pd.read_csv(logger.get_data_file(train_filename), skipinitialspace = True, 
                         parse_dates = ['DATETIME'], date_parser = dateparse, index_col = 'DATETIME')
  df_train.sort_index(inplace = True)
  logger.log("Finish reading training file", show_time = True)
  
  data = df_train.loc[:, df_train.columns != 'ATT_FLAG']
  labels = df_train['ATT_FLAG']
  outliers_idxs = pca_get_outliers(data, 15, logger)

  df_train_indexes = df_train.index.values[outliers_idxs]
  new_df = df_train.drop(df_train_indexes)