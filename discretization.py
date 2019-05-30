import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd


class PercentilesDiscretization():

  def __init__(self, quantiles, logger):
    
    self.quantiles = quantiles
    self.logger = logger
    self.discrete_signal = None


  def discretize_signal(self, signal):

    self.signal = signal
    self.discrete_signal, self.bins = pd.cut(
      signal, self.quantiles, retbins=True, labels = range(self.quantiles))
    self.logger.log("Percentile discretization using {} quantiles".format(self.quantiles))


  def visualize_discretization(self, start_time, end_time, title, axis_label, filename):

    int_to_mean_bins = {}
    for i in range(self.quantiles):
      int_to_mean_bins[i] = (self.bins[i] + self.bins[i+1]) / 2

    self.discrete_signal.replace(to_replace = int_to_mean_bins, inplace = True)

    sns.set()
    plt.figure(figsize=(12,6))
    ax = plt.subplot(111)

    signal_slice = self.signal.loc[start_time : end_time]
    discrete_signal_slice = self.discrete_signal.loc[start_time : end_time]

    plt.plot(signal_slice.index, signal_slice.values,  color = "blue", label = "Original signal")
    plt.stem(signal_slice.index, discrete_signal_slice.values, label = "Discrete point")
    plt.step(signal_slice.index, discrete_signal_slice.values, color = "gold", 
      label = "Discrete signal")

    for i in range(self.quantiles + 1):
      plt.axhline(y= self.bins[i], color = 'grey', linestyle = '--', linewidth = 1, 
        label = "Percentile boundary" if i == 1 else None)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(axis_label)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc = 'center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    plt.savefig(self.logger.get_output_file(filename), dpi = 120, bbox_inches='tight')


  def get_discrete_signal(self):

    if self.discrete_signal is None:
      self.logger.log("No signal previously discretize. Call discretize_signal before.")

    return self.discrete_signal


'''
signal = df_discrete.values.tolist()
a = zip(*[signal[i:] for i in range(3)])
from collections import Counter
Counter(a).items()
d = {k: v / len(a) for k, v in Counter(a).items()}
sum([v for k,v in d.items()])
'''