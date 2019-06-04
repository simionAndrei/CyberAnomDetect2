from sklearn.metrics import classification_report
import numpy as np


def predict_by_index(predict_index, true):
    predict = np.full(len(true), False)

    predict[predict_index] = True

    return print_scores(predict, true)


def print_scores(predict, true):
    classification = classification_report(true, predict, output_dict=True)['True']

    total = sum(predict)
    tp = sum(predict & true)
    fp = total - tp

    print("TP: {}\tFP: {}\tPrecision: {}\tRecall: {}\tF1: {}\tTotal T: {}\tTotal: {}".format(tp, fp, classification['precision'],
                                                                     classification['recall'],
                                                                     classification['f1-score'], total, len(predict)))

    return tp, fp, classification


if __name__ == "__main__":
    print_scores(np.array([False, True, True]), np.array([False, True, False]))
    predict_by_index(np.array([1,2]), np.array([False, True, False]))
