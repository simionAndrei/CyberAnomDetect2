from discretization import PercentilesDiscretization
from collections import Counter

import pandas as pd

class NGramsModel():

  def __init__(self, n, logger):

    self.n = n
    self.logger = logger
    self.thresholds = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.005]
    self.ngrams_probs = {}
    self.optim_thresholds = {}
    self._read_data()


  def _read_data(self):

    dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%y %H')
    train_filename = self.logger.config_dict['TRAIN_FILE']
    self.logger.log("Start reading training file {}...".format(train_filename))
    self.df_train = pd.read_csv(self.logger.get_data_file(train_filename), skipinitialspace = True, 
      parse_dates = ['DATETIME'], date_parser = dateparse, index_col = 'DATETIME')
    self.df_train.sort_index(inplace = True)
    self.logger.log("Finish reading training file", show_time = True)

    optim_filename = self.logger.config_dict['OPTIM_FILE']
    self.logger.log("Start reading optimization file {}...".format(optim_filename))
    self.df_optim = pd.read_csv(self.logger.get_data_file(optim_filename), skipinitialspace = True,
      parse_dates = ['DATETIME'], date_parser = dateparse, index_col = 'DATETIME')
    self.df_optim.sort_index(inplace = True)
    self.logger.log("Finish reading optimization file", show_time = True)

    test_filename = self.logger.config_dict['TEST_FILE']
    self.logger.log("Start reading testing file {}...".format(test_filename))
    self.df_test = pd.read_csv(self.logger.get_data_file(test_filename), skipinitialspace = True,
      parse_dates = ['DATETIME'], date_parser = dateparse, index_col = 'DATETIME')
    self.df_test.sort_index(inplace = True)
    self.logger.log("Finish reading test file", show_time = True)


  def compute_ngrams(self, signal):

    return zip(*[signal[i:] for i in range(self.n)])


  def fit_signal(self, signal_type, quantiles = 8):

    self.discretizator = PercentilesDiscretization(quantiles = quantiles, logger = self.logger)
    self.discretizator.discretize_signal(self.df_train[signal_type])
    discrete_signal = self.discretizator.get_discrete_signal()

    self.logger.log("Fit signal {} PercentileDiscretize with {} quantiles".format(
      signal_type, quantiles))

    ngrams = list(self.compute_ngrams(discrete_signal))
    self.ngrams_probs[signal_type] = {k: v / len(ngrams) for k, v in Counter(ngrams).items()}


  def compute_optimal_threshold(self, signal_type):

    self.discretizator.discretize_signal(self.df_optim[signal_type])
    discrete_signal = self.discretizator.get_discrete_signal()

    tp, fp = 0, 0
    optim_rate = {}
    for threshold in self.thresholds: 
      for idx, ngram in enumerate(self.compute_ngrams(discrete_signal)):
        crt_prob = self.ngrams_probs[signal_type].get(ngram, None)

        if crt_prob is None or crt_prob < threshold:
          #self.logger.log("Predicted anomaly")

          start_flag = self.df_optim.iloc[idx]['ATT_FLAG']
          end_flag   = self.df_optim.iloc[idx + self.n - 1]['ATT_FLAG'] 
          if int(start_flag) == 1 or int(end_flag) == 1:
            tp += 1
          else:
            fp += 1

      if fp == 0:
        precision = 1 if tp != 0 else 0
      else:
        #precision = tp / (tp + fp)
        precision = tp / fp

      precision = tp

      optim_rate[threshold] = precision
      #tp / (fp if fp != 0 else 1)

    print(optim_rate)
    optim_threshold = sorted(optim_rate.items(), key=lambda tup: tup[1])[-1][0]
    self.logger.log("Optimal threshold for {} is {}".format(signal_type, optim_threshold))
    self.optim_thresholds[signal_type] = optim_threshold


  def fit(self):

    for signal_type in self.df_train.columns.values[:-1]:
      self.fit_signal(signal_type = signal_type)
      self.compute_optimal_threshold(signal_type = signal_type)


  def predict(self):

    for signal_type in self.df_test.columns.values[:-1]:
      self.logger.log("Start predicting for signal {}".format(signal_type))
      self.predict_anomalies(signal_type)


  def predict_anomalies(self, signal_type):

    self.discretizator.discretize_signal(self.df_test[signal_type])
    discrete_signal = self.discretizator.get_discrete_signal()

    tp, fp = 0, 0
    for idx, ngram in enumerate(self.compute_ngrams(discrete_signal)):

      crt_prob = self.ngrams_probs[signal_type].get(ngram, None)

      if crt_prob is None or crt_prob < self.optim_thresholds[signal_type]:
        '''
        self.logger.log("Predicted anomaly at window:{}-{}".format(
          self.df_test.index[idx], self.df_test.index[idx + self.n - 1]))
        '''

        start_flag = self.df_optim.iloc[idx]['ATT_FLAG']
        end_flag   = self.df_optim.iloc[idx + self.n - 1]['ATT_FLAG']  
        if int(start_flag) == 1 or int(end_flag) == 1:
          tp += 1
        else:
          fp += 1

    self.logger.log("Signal {}: TP#{} / FP#{}".format(signal_type, tp, fp))

    return tp, fp

