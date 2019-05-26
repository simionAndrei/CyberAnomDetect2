from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from datetime import datetime

class MovingAverage():

  def __init__(self, window_size, logger):

    self.logger = logger
    self.window_size = window_size
    self.logger.log("Init moving average with window size {}".format(window_size))


  def load_signal(self, times, values, signal_str):

    self.train = np.array([values[i] for i in range(self.window_size)])
    self.test  = np.array([values[i] for i in range(self.window_size, times.shape[0])])
    self.preds = np.array([])
    self.signal_str = signal_str
    self.times = pd.to_datetime(times)

  def predict(self):
    
    for i, crt_obs in enumerate(self.test):
      y_pred = self.train[self.train.shape[0] - self.window_size : self.train.shape[0]].mean()
      self.preds = np.append(self.preds, y_pred)
      self.train = np.append(self.train, crt_obs)
      
      if i % 1000 == 0:
        self.logger.log("Sample#{}- Pred: {} / True: {}".format(i, y_pred, crt_obs))

    mse = mean_squared_error(self.test, self.preds)
    self.logger.log("MSE: {}".format(mse))


  def plot_prediction(self, start_time, stop_time, filename):

    start_time = datetime.strptime(start_time, '%Y-%m-%d %H')
    stop_time  = datetime.strptime(stop_time,  '%Y-%m-%d %H')

    sns.set()
    plt.figure(figsize=(12,5))
    ax = plt.subplot(111)

    plt.title("Moving average with window size {} for {}".format(self.window_size, self.signal_str))
    plt.xlabel("Time")
    plt.ylabel("Level (m)")

    selected_idx = np.where(np.logical_and(self.times > start_time, self.times < stop_time))[0]

    plt.plot(self.times[selected_idx], self.test[selected_idx], color = 'blue')
    plt.plot(self.times[selected_idx], self.preds[selected_idx], color = 'gold')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(["True signal", "Predicted signal"], loc = 'center left',
      bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    plt.savefig(self.logger.get_output_file(filename), dpi = 120, bbox_inches='tight')