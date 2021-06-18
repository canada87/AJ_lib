
# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████

import sys
sys.path.insert(0, 'C:/Users/Max Power/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
from AJ_models_regression import learning_reg
from AJ_models_classifier import learning_class

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

class work_functions:
    
    def time_series_regression(self, start, X_train, Y_train, X_test, Y_test = [], net_type = 'normal',  verbose_dl = 0):
        learn = learning_reg()
        #####################################################
        models = dict()
        models['deep learning '+net_type] = learn.get_deep_learning_model(input_dl = X_train.shape[1], output_dl = Y_train.shape[1], net_type = net_type)

        if len(Y_test)>0:
            models, history = learn.train_models(models, X_train, Y_train, epochs = 20, validation_data = (X_test, Y_test), batch_size = 64)
        else:
            models, history = learn.train_models(models, X_train, Y_train, epochs = 20, batch_size = 64)

        if verbose_dl == 1:
            learn.plot_history(history)

        X_to_predict = X_test.copy()
        if net_type == 'vgg':
            xvgg_test = X_test.copy()
            xvgg_test = xvgg_test.to_numpy()
            xvgg_test = xvgg_test.reshape(xvgg_test.shape[0], xvgg_test.shape[1], 1)
            X_to_predict = xvgg_test.copy()
        y_pred = models['deep learning '+net_type].predict(X_to_predict)
        #####################################################

        y_pred = pd.DataFrame(y_pred)
        y_pred.index = X_test.index

        mae = pd.DataFrame()
        if start == 'train':
            y_pred.columns = Y_test.columns
            y_predT = y_pred.T
            for col in list(y_predT.columns):
                mae[col] = [mean_absolute_error(Y_test.T[col], y_predT[col])]

        return y_pred, mae.mean(axis=1)

    def time_series_rolling_windows(self, start, x_train, y_train, x_test, y_test, rol_frame, net_type = 'normal'):
        learn = learning_reg()
        #####################################################
        models = dict()
        models['deep learning '+net_type] = learn.get_deep_learning_model(input_dl = x_train.shape[1], output_dl = rol_frame, net_type = net_type)
        models = learn.train_rolling_window2(models, x_train, y_train, rol_frame, epochs = 20)
        predict_matrix = learn.predict_rolling_window2(models, x_test, y_test.shape[1], rol_frame)

        mae = pd.DataFrame()
        if start == 'train':
            predict_matrix.columns = y_test.columns
            predict_matrix = predict_matrix.T
            for col in list(predict_matrix.columns):
                mae[col] = [mean_absolute_error(y_test.T[col], predict_matrix[col])]

        return predict_matrix, mae.mean(axis=1)

    def classification(self, x_train, y_train, X_test, Y_test, net_type = 'normal', loss_type = 'binary', verbose_plot = 0, verbose_dl = 0, verbose = 0):
        learn = learning_class()
        models = dict()
        models = learn.get_models(['RandomForestClassifier', 'MLPClassifier', 'GradientBoostingClassifier'])
        models['deep learning '+net_type+' '+loss_type] = learn.get_deep_learning_model(x_train.shape[1], net_type = net_type, loss_type = loss_type)
        models, history = learn.train_models(models, x_train, y_train, epochs = 10, validation_data = (X_test, Y_test), batch_size = 64)
        if verbose_dl == 1:
            learn.plot_history(history)
        num_class = int(Y_test.unique().shape[0])
        y_prob_dict, y_pred_matrix = learn.prob_matrix_generator(models, X_test, num_class)
        print(y_pred_matrix)
        score = learn.score_models(models, Y_test, y_prob_dict, y_pred_matrix)
        if verbose == 1:
            print(score)
        if verbose_plot == 1:
            learn.plot_roc_curve(y_prob_dict, Y_test)
            learn.score_accuracy_recall(y_pred_matrix, Y_test, verbose = 1)
            learn.plot_precision_recall(y_prob_dict, Y_test)
        return score, y_pred_matrix

    def multi_classification_non_mutualy_exlusive(x_train, y_train, x_test, y_test, verbose_plot = 0, verbose_dl = 0, verbose = 0):
        learn = learning_class()
        models = dict()
        models = learn.get_models(['RandomForestClassifier', 'DecisionTreeClassifier'])
        # models = learn.get_models(['RandomForestClassifier', 'MLPClassifier', 'GradientBoostingClassifier'])
        print(x_train.shape)
        num_classes = y_train.shape[1]

        net_type = 'vgg'
        loss_type = 'multisparse'
        models['deep learning '+net_type+' '+loss_type] = learn.get_deep_learning_model(x_train.shape[1], net_type = net_type, loss_type = loss_type, num_classes = num_classes)
        models, hystory = learn.train_models(models, x_train, y_train, epochs = 50, validation_data = (x_test, y_test))

        if verbose_dl == 1:
            learn.plot_history(hystory)

        num_class = int(y_test.shape[1])
        y_prob_dict, y_pred_matrix = learn.prob_matrix_generator(models, x_test, num_class, multi_class = True)
        # print(y_pred_matrix)
        # print(y_prob_dict)
        df_score = pd.DataFrame()
        for clas in learn.classes:
            y_test_class, y_prob_dict_class, y_pred_matrix_class = learn.from_multilabel_to_single(y_test, y_prob_dict, y_pred_matrix, clas)
            # print(y_pred_matrix_class)
            score = learn.score_models(y_test_class, y_prob_dict_class, y_pred_matrix_class)
            df_score[clas] = score.loc['accuracy', :]
            if verbose == 1:
                print(score)
            if verbose_plot == 1:
                learn.score_accuracy_recall(y_pred_matrix_class, y_test_class, verbose = 1)
        df_score.index = y_pred_matrix.keys()
        return df_score.T, y_pred_matrix
