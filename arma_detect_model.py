from operator import itemgetter
from arma import *


class ArmaDetectModel:

    def __init__(self, logger):

        self.logger = logger
        self.coefficients = np.arange(0.1, 50, 0.1)

        self.params = [[5, 1, 0, 'L_T1'], [5, 1, 0, 'L_T2'], [5, 1, 0, 'L_T3'], [5, 1, 0, 'L_T4'], [5, 1, 0, 'L_T5'],
                       [5, 1, 0, 'L_T6'], [5, 1, 0, 'L_T7'], [5, 1, 0, 'F_PU1'], [5, 1, 0, 'F_PU2'],
                       [5, 1, 0, 'S_PU2'], [5, 1, 0, 'F_PU4'],
                       [5, 1, 0, 'S_PU4'], [4, 0, 0, 'F_PU6'],
                       [4, 0, 0, 'S_PU6'], [5, 1, 0, 'F_PU7'], [5, 1, 0, 'S_PU7'], [5, 1, 0, 'F_PU8'],
                       [5, 1, 0, 'S_PU8'], [5, 1, 0, 'F_PU10'],
                       [5, 1, 0, 'S_PU10'], [0, 1, 0, 'F_PU11'], [0, 1, 0, 'S_PU11'], [5, 1, 0, 'F_V2'],
                       [4, 1, 0, 'S_V2'], [5, 1, 0, 'P_J280'], [5, 1, 0, 'P_J269'], [5, 1, 0, 'P_J300'],
                       [5, 1, 0, 'P_J256'], [5, 1, 0, 'P_J289'], [5, 1, 0, 'P_J415'], [5, 1, 0, 'P_J302'],
                       [5, 1, 0, 'P_J306'], [5, 1, 0, 'P_J307'], [5, 1, 0, 'P_J317'], [5, 1, 0, 'P_J14'],
                       [5, 1, 0, 'P_J422']]

        self.optim_thresholds = {}
        self.models = []

        self._read_data()

    def _read_data(self):

        dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%y %H')
        train_filename = self.logger.config_dict['TRAIN_FILE']
        self.logger.log("Start reading training file {}...".format(train_filename))
        self.df_train = pd.read_csv(self.logger.get_data_file(train_filename), skipinitialspace=True,
                                    parse_dates=['DATETIME'], date_parser=dateparse, index_col='DATETIME').asfreq('H')
        self.df_train.sort_index(inplace=True)
        self.logger.log("Finish reading training file", show_time=True)

        optim_filename = self.logger.config_dict['OPTIM_FILE']
        self.logger.log("Start reading optimization file {}...".format(optim_filename))
        self.df_optim = pd.read_csv(self.logger.get_data_file(optim_filename), skipinitialspace=True,
                                    parse_dates=['DATETIME'], date_parser=dateparse, index_col='DATETIME').asfreq('H')
        self.df_optim.sort_index(inplace=True)
        self.logger.log("Finish reading optimization file", show_time=True)

        self.start_optim = self.df_optim.index[0]
        self.end_optim = self.df_optim.index[-1]

        self.optim_attacks_location = np.where(self.df_optim['ATT_FLAG'] == 1)

        test_filename = self.logger.config_dict['TEST_FILE']
        self.logger.log("Start reading testing file {}...".format(test_filename))
        self.df_test = pd.read_csv(self.logger.get_data_file(test_filename), skipinitialspace=True,
                                   parse_dates=['DATETIME'], date_parser=dateparse, index_col='DATETIME').asfreq('H')
        self.df_test.sort_index(inplace=True)
        self.logger.log("Finish reading test file", show_time=True)

        self.test_attacks_location = np.where(self.df_test['ATT_FLAG'] == 1)

    def _compute_stats(self, series, signal_name, model):
        true_points = np.array(series[signal_name])

        predict_points = np.array(
            model.predict(start=0, end=-1))

        std_ma = series[signal_name].rolling(24).mean().std()

        se = true_points - predict_points
        se **= 2

        mse = se.mean()

        return predict_points, true_points, std_ma, se, mse

    def _compute_optimal_threshold(self, model, signal_name):
        (predict_points, true_points, std_ma, se, mse) = self._compute_stats(self.df_optim, signal_name, model)

        best_ratio = -1
        optim_coefficient = None

        for coefficient in self.coefficients:
            # compute the threshold value
            threshold = coefficient * std_ma + mse

            # calculate predictions
            predictions = predict_points > threshold

            total_estimated_attacks = predictions.sum()

            # tp is the sum of values in the attack positions
            tp = (predictions[self.optim_attacks_location]).sum()

            fp = total_estimated_attacks - tp
            # fp = (predictions * normal_points).sum()

            if fp == 0:
                ratio = 1 if tp != 0 else 0
            else:
                ratio = tp / fp

            if ratio > best_ratio:
                best_ratio = ratio
                optim_coefficient = coefficient

        self.logger.log("Optimal threshold for {} is {}".format(signal_name, optim_coefficient))
        self.optim_thresholds[signal_name] = optim_coefficient

        return [optim_coefficient, best_ratio]

    def fit(self):

        for [p, q, d, signal_name] in self.params:
            model = create_model(self.df_optim[signal_name], p, q, d)
            [coefficient, ratio] = self._compute_optimal_threshold(model, signal_name)

            self.models.append((model, coefficient, signal_name, [p, q, d, signal_name], ratio))

        self.models.sort(key=itemgetter(4), reverse=True)

        # take the top 20 models
        # ideally this constant should be deduced via cross-validation
        self.models = self.models[:20]


    def predict(self):

        results = []
        total_predictions = np.full(len(self.df_test), False)
        for (model, coefficient, signal_name, [p, q, d, signal_name], ratio) in self.models:
            try:
                model = create_model(self.df_test[signal_name], p, q, d)
            except:
                continue
            (predict_points, true_points, std_ma, se, mse) = self._compute_stats(self.df_test, signal_name, model)

            # compute the threshold value
            threshold = coefficient * std_ma + mse

            # calculate predictions
            predictions = predict_points > threshold

            total_estimated_attacks = predictions.sum()

            # tp is the sum of values in the attack positions
            tp = (predictions[self.test_attacks_location]).sum()

            fp = total_estimated_attacks - tp

            if fp == 0:
                ratio = 1 if tp != 0 else 0
            else:
                ratio = tp / fp

            result = "Signal {}: TP#{} / FP#{} / RATIO#{}".format(signal_name, tp, fp, ratio)

            results.append([ratio, result, predictions])
            self.logger.log(result)

            total_predictions = total_predictions + predictions

        total_estimated_attacks = total_predictions.sum()

        # tp is the sum of values in the attack positions
        tp = (total_predictions[self.test_attacks_location]).sum()

        fp = total_estimated_attacks - tp

        self.logger.log("TOTAL: TP#{} / FP#{}".format(tp, fp))

        results.sort(key=itemgetter(0), reverse=True)

        self.logger.log("Best signals {}".format(results))

        selected_predictions = np.full(len(self.df_test), False)

        for [_, _, prediction] in results[:5]:
            selected_predictions = selected_predictions + prediction

        selected_estimated_attacks = selected_predictions.sum()

        # tp is the sum of values in the attack positions
        tp = (selected_predictions[self.test_attacks_location]).sum()

        fp = selected_estimated_attacks - tp

        # This is purely theoretical because we need to know the labels to deduce this :)
        self.logger.log("BEST COMBINATION: TP#{} / FP#{}".format(tp, fp))




if __name__ == "__main__":
    logger = Logger(show=True, html_output=True, config_file="config.txt")
    model = ArmaDetectModel(logger)
    model.fit()
    model.predict()
