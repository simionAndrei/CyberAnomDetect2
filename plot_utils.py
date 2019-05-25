from bokeh.plotting import figure, output_file, save

from bokeh.models import CustomJS, ColumnDataSource, Select, CheckboxGroup
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column


def make_water_tanks_plot(data_df):

  columns = ['L_T' + str(i) for i in range(1, 8)]
  max_y = max([data_df[col].max() for col in columns])

  source = ColumnDataSource({
    'x_t1' : data_df.index, 'y_t1' : data_df['L_T1'],
    'x_t2' : data_df.index, 'y_t2' : data_df['L_T2'],
    'x_t3' : data_df.index, 'y_t3' : data_df['L_T3'],
    'x_t4' : data_df.index, 'y_t4' : data_df['L_T4'],
    'x_t5' : data_df.index, 'y_t5' : data_df['L_T5'],
    'x_t6' : data_df.index, 'y_t6' : data_df['L_T6'],
    'x_t7' : data_df.index, 'y_t7' : data_df['L_T7'],
  })

  p = figure(plot_width=1000, plot_height=700,
      title="Water tanks", tools = "wheel_zoom,box_zoom,save",
      x_axis_type="datetime", y_range=[0, max_y + 1],
      x_axis_label = "Time",
      y_axis_label = "Water level (m)")

  p_t1 = p.line('x_t1', 'y_t1', source=source, color='red', 
    alpha=0.9, line_width = 2, legend = "Tank1")
  p_t2 = p.line('x_t2', 'y_t2', source=source, color='yellow',
    alpha=0.9, line_width = 2, legend = "Tank2")
  p_t3 = p.line('x_t3', 'y_t3', source=source, color='blue',
    alpha=0.9, line_width = 2, legend = "Tank3")
  p_t4 = p.line('x_t4', 'y_t4', source=source, color='black',
    alpha=0.9, line_width = 2, legend = "Tank4")
  p_t5 = p.line('x_t5', 'y_t5', source=source, color='green',
    alpha=0.9, line_width = 2, legend = "Tank5")
  p_t6 = p.line('x_t6', 'y_t6', source=source, color='salmon',
    alpha=0.9, line_width = 2, legend = "Tank6")
  p_t7 = p.line('x_t7', 'y_t7', source=source, color='navy',
    alpha=0.9, line_width = 2, legend = "Tank7")

  p.legend.location = (1000, 420)

  tanks = ['Tank ' + str(i) for i in range(1, 8)]
  checkboxes = CheckboxGroup(labels=tanks, active=[i for i in range(7)])
  callback = CustomJS(args=dict(p_t1=p_t1, p_t2=p_t2, p_t3=p_t3, p_t4=p_t4, 
    p_t5=p_t5, p_t6=p_t6, p_t7=p_t7, p = p), 
    code="""
    f = cb_obj.active;
    p_t1.visible = false;
    p_t2.visible = false;
    p_t3.visible = false;
    p_t4.visible = false;
    p_t5.visible = false;
    p_t6.visible = false;
    p_t7.visible = false;
    if (f.includes(0)) { p_t1.visible = true; }
    if (f.includes(1)) { p_t2.visible = true; }
    if (f.includes(2)) { p_t3.visible = true; }
    if (f.includes(3)) { p_t4.visible = true; }
    if (f.includes(4)) { p_t5.visible = true; }
    if (f.includes(5)) { p_t6.visible = true; }
    if (f.includes(6)) { p_t7.visible = true; }
    p.legend.location = "top_right";
    """
  )
  checkboxes.js_on_click(callback)

  output_file("tanks.html")
  save(column(checkboxes, p))



from logger import Logger

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = Logger(show = True, html_output = True, config_file = "config.txt")

dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%y %H')
train_filename = logger.config_dict['TRAIN_FILE']
df_train = pd.read_csv(logger.get_data_file(logger.config_dict['TRAIN_FILE']), 
                       parse_dates = ['DATETIME'], date_parser = dateparse, index_col = 'DATETIME')
df_train.sort_index(inplace = True)

df_train_days = df_train.groupby(pd.Grouper(freq='D')).mean()
from plot_utils import make_water_tanks_plot
make_water_tanks_plot(df_train_days)