import sys
sys.path.insert(0, 'C:/Users/Max Power/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
sys.path.insert(0, 'C:/Users/ajacassi/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
from AJ_draw import disegna as ds
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import pandas as pd
import numpy as np
import seaborn as sns
import json

from tensorflow import keras #funziona con tensorflow-gpu==1.14.0

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error
from xgboost import XGBClassifier

from scipy import interp

from sklearn.preprocessing import label_binarize

from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score, accuracy_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

class learning_class:
    def __init__ (self, SEED = 222):
        self.SEED = SEED

    def onehotencoding_numerical_vector(self, vector):
        #trasforma il vettore dei target in un onehot encoded
        if len(vector.unique()) > 2:
            vector_ohe = label_binarize(vector, classes=np.sort(vector.unique()))
        else:
            n_values = len(vector.unique())
            onehot_vector = np.zeros([len(vector), n_values])
            onehot_vector[np.arange(len(vector)), vector] = 1
            vector_ohe = onehot_vector.astype(int)
        return vector_ohe



        # ███    ███  ██████  ██████  ███████ ██      ███████
        # ████  ████ ██    ██ ██   ██ ██      ██      ██
        # ██ ████ ██ ██    ██ ██   ██ █████   ██      ███████
        # ██  ██  ██ ██    ██ ██   ██ ██      ██           ██
        # ██      ██  ██████  ██████  ███████ ███████ ███████




    def get_models(self, list_chosen):
        """Generate a library of base learners.
        :param list_chosen: list with the names of the models to load
        :return: models, a dictionary with as index the name of the models, as elements the models"""

        # dtc = DecisionTreeClassifier(max_depth=3, random_state=self.SEED)
        # nb = GaussianNB()
        # svc = SVC(C=100, probability=True, gamma = 'scale')
        # knn = KNeighborsClassifier(n_neighbors=24, leaf_size=70, p=1)
        # lr = LogisticRegression(C=100, random_state=self.SEED, solver='lbfgs', max_iter=8000)
        nn = MLPClassifier((80, 10), early_stopping=False, random_state=self.SEED)
        # gb = GradientBoostingClassifier(n_estimators=100, random_state=self.SEED)
        # rf = RandomForestClassifier(n_estimators=10, max_features=3, random_state=self.SEED)
        xgb = XGBClassifier(random_state=self.SEED, n_estimators = 300, max_depth = 4, learning_rate= 0.068)

        dtc = DecisionTreeClassifier(random_state=self.SEED)
        nb = GaussianNB()
        svc = SVC(probability=True)
        knn = KNeighborsClassifier()
        lr = LogisticRegression(random_state=self.SEED,  solver='lbfgs', max_iter=8000)
        # nn = MLPClassifier(random_state=self.SEED)
        gb = GradientBoostingClassifier(random_state=self.SEED)
        rf = RandomForestClassifier(random_state=self.SEED)
        # xgb = XGBClassifier(random_state=self.SEED)


        models_temp = {
                  'LogisticRegression': lr,
                  'SVC': svc,
                  'KNeighborsClassifier': knn,
                  'MLPClassifier': nn,
                  'DecisionTreeClassifier': dtc,
                  'RandomForestClassifier': rf,
                  'GradientBoostingClassifier': gb,
                  'naive bayes': nb,
                  'XGBClassifier':xgb
                  }

        models = dict()
        for model in list_chosen:
            if model in models_temp:
                models[model] = models_temp[model]
        return models

    def get_deep_learning_model(self, input_dl, active = 'relu', num_classes = 2, net_type = 'normal', loss_type = 'sparse'):
        ''' generate the deep learning model'''

        if net_type == 'normal':

            model = keras.Sequential()
            # model.add(keras.layers.Dense(int(input_dl/2), activation=active, input_dim = input_dl))
            # model.add(keras.layers.Dense(int(input_dl/10), activation = active))
            # model.add(keras.layers.Dense(int(input_dl/100), activation = active))

            model.add(keras.layers.Dense(10, activation=active, input_dim = input_dl))
            model.add(keras.layers.Dense(5, activation = active))

            if loss_type == 'sparse':
            # predice da 2 a n classi, le classi NON devono essere one hot encoading
            # siccome il layer finale ha una softmax functino viene dato come risultato solo il piu probabile
            # serve quando si hanno molte classi ma ogni dato appartiene ad una e una sola classe
                model.add(keras.layers.Dense(num_classes, activation = 'softmax'))
                model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

            # predice da 2 a n classi, le classi devono trovarsi in one hot encoading visto che ogni classe ha il suo nodo di uscita ed e' abbinato ad una signoide che da valori di uscita tra 0 e 1,
            # in pratica si ha su ogni nodo la probabilita di avere quella classe
            # serve per quando si hanno piu classi che possono comparire per lo stesso dato (multiclasse)
            elif loss_type == 'multisparse':
                model.add(keras.layers.Dense(num_classes, activation='sigmoid'))
                model.compile(loss='mse', optimizer = 'adam', metrics = ['accuracy'])

            # in presenza di 2 classi messe in un unico vettore 0 e' una classe 1 e' l'altra.
            # ogni dato puo appartenere ad una sola classe
            elif loss_type == 'binary':
                model.add(keras.layers.Dense(1, activation = 'sigmoid'))
                model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

        elif net_type == 'vgg':
            def vgg_block(layer_in, n_filters, n_conv):
                for _ in range(n_conv):
                    layer_in = keras.layers.Conv1D(n_filters, 3, padding = 'same', activation = 'relu')(layer_in)
                layer_in = keras.layers.MaxPooling1D(2, strides = 2)(layer_in)
                return layer_in
            visible = keras.layers.Input(shape = (input_dl,1))
            layer = vgg_block(visible, 64, 4)
            layer = keras.layers.Dropout(0.1)(layer)
            layer = vgg_block(layer, 128, 4)
            layer = keras.layers.Dropout(0.2)(layer)
            layer = vgg_block(layer, 256, 8)
            layer = keras.layers.Dropout(0.3)(layer)
            flat1 = keras.layers.Flatten()(layer)
            hidden1 = keras.layers.Dense(512, activation = 'relu')(flat1)
            hidden1 = keras.layers.Dropout(0.4)(hidden1)
            if loss_type == 'sparse':
                output = keras.layers.Dense(num_classes, activation = 'softmax')(hidden1)
                model = keras.models.Model(inputs = visible, outputs = output)
                model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
            elif loss_type == 'binary':
                output = keras.layers.Dense(1, activation = 'sigmoid')(hidden1)
                model = keras.models.Model(inputs = visible, outputs = output)
                model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

        return model


        # ████████ ██████   █████  ██ ███    ██ ██ ███    ██  ██████
        #    ██    ██   ██ ██   ██ ██ ████   ██ ██ ████   ██ ██
        #    ██    ██████  ███████ ██ ██ ██  ██ ██ ██ ██  ██ ██   ███
        #    ██    ██   ██ ██   ██ ██ ██  ██ ██ ██ ██  ██ ██ ██    ██
        #    ██    ██   ██ ██   ██ ██ ██   ████ ██ ██   ████  ██████




    def train_models(self, models, xtrain, ytrain, epochs = None, validation_data = None, shuffle = True, batch_size = 32):
        '''training function
        :param models: dictionary with as index the name of the models, as elements the models
        :param xtrain: matrix with the features, pandas
        :param ytrain: array with the targets, pandas (it is NOT onehot encoded)
        :param epochs: number of epochs if deep learning is used
        :param validation_data: data used to validate if deep learning or XGBClassifier are used, pandas or numpy in the same format of the training.
                                If deep learning is used validation_data = (xval, yval)
                                If XGBClassifier is used, training data can be passed as well to generate a complete report validation_data = [(xtrain, ytrain),(xval, yval)]
        :param shuffle: Boolean, used to shuffle the data before the training if deep learning is used
        :return: models, a dictionary with as index the name of the models, as elements the models after the training
        :return: fitModel, scoring hystory of the deep learning over epochs or XGBClassifier'''
        fitModel = dict()
        for i, (name_model, model) in enumerate(models.items()):
            if name_model == 'deep learning normal sparse' or name_model == 'deep learning normal binary' or name_model == 'deep learning normal multisparse':
                fitModel[name_model] = model.fit(xtrain, ytrain, epochs = epochs, verbose = 1, validation_data= validation_data, shuffle = shuffle, batch_size = batch_size)
            elif name_model == 'deep learning vgg sparse' or name_model == 'deep learning vgg binary':
                xvgg = xtrain.copy()
                xvgg = xvgg.to_numpy()
                xvgg = xvgg.reshape(xvgg.shape[0], xvgg.shape[1], 1)
                validation_data_new = None
                if validation_data:
                    xvgg_val = validation_data[0]
                    xvgg_val = xvgg_val.to_numpy()
                    xvgg_val = xvgg_val.reshape(xvgg_val.shape[0], xvgg_val.shape[1], 1)
                    validation_data_new = (xvgg_val,validation_data[1])
                fitModel[name_model] = model.fit(xvgg, ytrain, epochs = epochs, verbose = 1, validation_data= validation_data_new, shuffle = shuffle, batch_size = batch_size)
            elif name_model == 'XGBClassifier' and validation_data:
                model.fit(xtrain, ytrain, eval_metric=["error", "logloss"], eval_set=validation_data, verbose=False, early_stopping_rounds = epochs)
                fitModel[name_model] = model.evals_result()
            else:
                model.fit(xtrain, ytrain)
        return models, fitModel

    def train_hyperparameters(self, models, xtrain, ytrain, dict_grid, cv = 5, filename = 'param_file', type = 'random'):
        '''Training function for the hyperparameters search, valid only for the standard models
        :param models: dictionary with as index the name of the models, as elements the models
        :param xtrain: matrix with the features, pandas
        :param ytrain: array with the targets, pandas (it is NOT onehot encoded)
        :param dict_grid: dictionary with the list of the paramters and the limits for the search
        :param cv: number of kfold (int), or cv splitter
        :param filename: str, file name of the list with the resutls
        :param type: 'random' = RandomizedSearchCV, 'grid' = 'GridSearchCV'
        :return: dict_best_param, dictionary with the best parameters for each model in the models
        '''
        dict_best_param = {}
        for i, (name_model, model) in enumerate(models.items()):
            if type == 'random':
                model_random = RandomizedSearchCV(estimator=model, param_distributions=dict_grid[name_model], n_iter=100, cv=cv, verbose=2, random_state=1, n_jobs=-1)
            else:
                model_random = GridSearchCV(estimator=model, param_grid=dict_grid[name_model], cv=cv, verbose=2, n_jobs=-1)
            model_random.fit(xtrain, ytrain)
            dic_h = model_random.best_params_
            dict_best_param[name_model] = dic_h
        with open(filename, 'w') as f:
            json.dump(dict_best_param, f)
        return dict_best_param

    def calibration_model(self, models, x_train, y_train):
        '''calibration
        Some models can benefit from a calibration process after the training, it would normalize the probabilities
        :param models: dictionary with as index the name of the models, as elements the models
        :param xtrain: matrix with the features, pandas
        :param ytrain: array with the targets, pandas (it is NOT onehot encoded)
        :return: calibrator, a dictionary with as index the name of the models, as elements the models after the training
        '''
        calibrator = dict()
        for i, (name_model, model) in enumerate(models.items()):
            calibrator[name_model] = CalibratedClassifierCV(model, cv='prefit')
            calibrator[name_model].fit(x_train, y_train)
        return calibrator




        # ███████  ██████  ██████  ███████  ██████  █████  ███████ ████████
        # ██      ██    ██ ██   ██ ██      ██      ██   ██ ██         ██
        # █████   ██    ██ ██████  █████   ██      ███████ ███████    ██
        # ██      ██    ██ ██   ██ ██      ██      ██   ██      ██    ██
        # ██       ██████  ██   ██ ███████  ██████ ██   ██ ███████    ██




    def prob_matrix_generator(self, models, xtest, num_classes):
        '''generate the prediction and the probabilities with all the models in the list and add all of them to the same matrix, adding the average prediciton (ensamble)
        :param models: dictionary with as index the name of the models, as elements the models
        :param xtest: matrix with the features, pandas
        :param num_classes: number of classes, int
        :return: y_pred_matrix.shape[0] = xtest.shape[0], y_pred_matrix.shape[1] = len(models) + 1, Pandas matrix with the predicion for all the models and the average for the ensamble
        :return: y_prob_dict[model].shape[0] = xtest.shape[0], y_prob_dict[model].shape[1] = ytest.unique().shape[0], dictionary with a pandas matrix for each model, the pandas matrix represent the probability of success for each classes

        it behaves differently if 'multispare' is selected, the function return imediatly with as y_prob_dict the probabilities for each classes and Y_pred the predictino for each classes.
        It cannot be ensamble with the other models since 'multispare' can present multiple classes at the same time as valible prediction
        '''

        y_pred_matrix = pd.DataFrame()
        y_prob_dict = {}
        for i, (name_model, model) in enumerate(models.items()):
            if name_model == 'deep learning normal sparse':
                y_prob_dict[name_model] = pd.DataFrame(model.predict_proba(xtest))
                y_pred_matrix[name_model] = model.predict_classes(xtest)

            elif name_model == 'deep learning normal multisparse':
                y_prob_dict[name_model] = pd.DataFrame(model.predict(xtest))
                df_pred = pd.DataFrame()
                for col in y_prob_dict[name_model].columns:
                    df_pred[col] = round(y_prob_dict[name_model][col])
                return y_prob_dict[name_model], df_pred

            elif name_model == 'deep learning normal binary':
                list_temp = model.predict_proba(xtest)
                list_temp = list_temp.reshape(1,-1)[0]
                df_temp = pd.DataFrame()
                df_temp[0] = 1-list_temp
                df_temp[1] = list_temp
                y_prob_dict[name_model] = df_temp
                y_pred_matrix[name_model] = model.predict_classes(xtest)[:,0]#.reshape(1,-1)[0]

            elif name_model == 'deep learning vgg sparse':
                xvgg = xtest.copy()
                xvgg = xvgg.to_numpy()
                xvgg = xvgg.reshape(xvgg.shape[0], xvgg.shape[1], 1)
                y_prob_dict[name_model] = pd.DataFrame(model.predict(xvgg))
                print(y_prob_dict[name_model])
                y_pred_matrix[name_model] = np.argmax(model.predict(xvgg),axis=1)

            elif name_model == 'deep learning vgg binary':
                xvgg = xtest.copy()
                xvgg = xvgg.to_numpy()
                xvgg = xvgg.reshape(xvgg.shape[0], xvgg.shape[1], 1)
                list_temp = model.predict(xvgg)
                list_temp = list_temp.reshape(1,-1)[0]
                df_temp = pd.DataFrame()
                df_temp[0] = 1-list_temp
                df_temp[1] = list_temp
                y_prob_dict[name_model] = df_temp
                y_pred_matrix[name_model] = np.argmax(df_temp.to_numpy(), axis = 1)

            else:
                y_prob_dict[name_model] = pd.DataFrame(model.predict_proba(xtest))
                y_pred_matrix[name_model] = model.predict(xtest)

        Prob_ens = np.zeros((xtest.shape[0], num_classes))
        Prob_ens = pd.DataFrame(Prob_ens)
        for name in y_prob_dict:
            Prob_ens = y_prob_dict[name] + Prob_ens
        Prob_ens = Prob_ens/len(y_prob_dict)
        y_prob_dict['Ensamble'] = Prob_ens
        y_pred_matrix['Ensamble'] = round(y_pred_matrix.mean(axis=1))
        y_pred_matrix['Ensamble'] = y_pred_matrix['Ensamble'].astype(int)
        return y_prob_dict, y_pred_matrix


        # ███████  ██████  ██████  ██████  ███████
        # ██      ██      ██    ██ ██   ██ ██
        # ███████ ██      ██    ██ ██████  █████
        #      ██ ██      ██    ██ ██   ██ ██
        # ███████  ██████  ██████  ██   ██ ███████




    def calibration_plot(self, y_prob_dict, y_true):
        '''When performing classification one often wants to predict not only the class label, but also the associated probability.
        This probability gives some kind of confidence on the prediction. The first figure shows the estimated probabilities obtained.
        The dotted diagonal represent the perfect calibration.
        :param y_prob_dict: dictionary of pandas matrices with the probability resutls for all classes (it is produced by the prob_matrix_generator function)
        :param y_true: array with the targets, pandas (it is NOT onehot encoded)
        '''
        y_true_single = self.onehotencoding_numerical_vector(y_true)
        # y_true_single = label_binarize(y_true, classes=np.sort(y_true.unique()))
        n_classes = len(y_true_single[0,:])
        cm = [plt.cm.rainbow(i) for i in np.linspace(0, 1.0, n_classes + 1)]
        j = 0
        for model in y_prob_dict:
            ds().nuova_fig(101+j)
            ds().titoli(titolo='calibration plot '+str(model), xtag='mean predicted value', ytag='fraction of positives')
            fop = dict()
            mpv = dict()
            for n_class in range(n_classes):
                fop[n_class], mpv[n_class] = calibration_curve(y_true_single[:,n_class], y_prob_dict[model].values[:,n_class], n_bins=10)
                ds().dati(mpv[n_class], fop[n_class], colore=cm[n_class+1], descrizione=str(n_class))
                ds().dati(mpv[n_class], fop[n_class], scat_plot ='scat', larghezza_riga =15, colore=cm[n_class+1])
            plt.plot([0,1],[0,1], linestyle = '--')
            ds().legenda()
            ds().porta_a_finestra()
            j = j + 1

    def score_models(self, y_true, y_prob_dict, y_pred_matrix):
        '''
        generate the score with the true target
        :param y_true: array with the targets, pandas (it is NOT onehot encoded)
        :param y_prob_dict: dictionary of pandas matrices with the probability resutls for all classes (it is produced by the prob_matrix_generator function)
        :param y_pred_matrix: pandas matrix with the predicted class for each model as column, the prediction comes in the same format of the target (it is NOT onehot encoded)
        :return: df_score.shape[1] = num of models + 1, df_score.shape[0] = (accuracy, mae, roc score)
        '''

        #trova quanti classi ci sono
        num_class = y_prob_dict[list(y_prob_dict.keys())[0]].shape[1]

        vet_nomi = []
        vet_accuracy = []
        matr_score = np.zeros((y_pred_matrix.shape[1], num_class))
        vet_mae = []

        #trasforma il vettore dei target in un onehot encoded
        y_test_roc = self.onehotencoding_numerical_vector(y_true)

        for i, name_model in enumerate(y_pred_matrix.columns.tolist()):
            accuracy = accuracy_score(y_true, y_pred_matrix[name_model])
            mae = mean_absolute_error(y_true, y_pred_matrix[name_model])

            j=0
            for col in y_prob_dict[name_model].columns:
                matr_score[i, j] = round(roc_auc_score(y_test_roc[:, j], y_prob_dict[name_model][col]),3)
                j = j + 1

            vet_nomi.append(name_model)
            vet_accuracy.append(round(float(accuracy),3))
            vet_mae.append(round(mae,3))

        score_data = pd.DataFrame()
        score_data['model'] = vet_nomi
        score_data['accuracy'] = vet_accuracy
        score_data['mae'] = vet_mae
        for i in range(len(y_true.unique())):
            score_data['roc score class ' + str(np.sort(y_true.unique())[i])] = matr_score[:, i]
        return score_data.T

    # def score_models_old(self, models, y_true, y_prob_dict, y_pred_matrix):
    #     '''
    #     generate the score with the true target
    #     :param models: dictionary with as index the name of the models, as elements the models
    #     :param y_true: array with the targets, pandas (it is NOT onehot encoded)
    #     :param y_prob_dict: dictionary of pandas matrices with the probability resutls for all classes (it is produced by the prob_matrix_generator function)
    #     :param y_pred_matrix: pandas matrix with the predicted class for each model as column, the prediction comes in the same format of the target (it is NOT onehot encoded)
    #     :return: df_score.shape[1] = num of models + 1, df_score.shape[0] = (accuracy, mae, roc score)
    #     '''
    #
    #     #trova quanti classi ci sono
    #     num_class = y_prob_dict[list(y_prob_dict.keys())[0]].shape[1]
    #
    #     vet_nomi = []
    #     vet_accuracy = []
    #     matr_score = np.zeros((len(models), num_class))
    #     vet_mae = []
    #
    #     #trasforma il vettore dei target in un onehot encoded
    #     y_test_roc = self.onehotencoding_numerical_vector(y_true)
    #
    #     for i, (name_model, model) in enumerate(models.items()):
    #         accuracy = accuracy_score(y_true, y_pred_matrix[name_model])
    #         mae = mean_absolute_error(y_true, y_pred_matrix[name_model])
    #
    #         j=0
    #         for col in y_prob_dict[name_model].columns:
    #             matr_score[i, j] = round(roc_auc_score(y_test_roc[:, j], y_prob_dict[name_model][col]),3)
    #             j = j + 1
    #
    #         vet_nomi.append(name_model)
    #         vet_accuracy.append(round(float(accuracy),3))
    #         vet_mae.append(round(mae,3))
    #
    #     score_data = pd.DataFrame()
    #     score_data['model'] = vet_nomi
    #     score_data['accuracy'] = vet_accuracy
    #     score_data['mae'] = vet_mae
    #     for i in range(len(y_true.unique())):
    #         score_data['roc score class ' + str(np.sort(y_true.unique())[i])] = matr_score[:, i]
    #
    #     vet_ave = []
    #     for i in range(len(models)):
    #         vet_ave.append(round(matr_score[i, :].mean(),3))
    #     score_data['roc score average'] = vet_ave
    #
    #     score_data = score_data.T
    #     ens_score = np.zeros(num_class)
    #     j = 0
    #     for col in y_prob_dict['Ensamble'].columns:
    #         ens_score[j] = round(roc_auc_score(y_test_roc[:, j], y_prob_dict['Ensamble'][col]),3)
    #         j = j + 1
    #
    #     ens_score_ave = round(ens_score.mean(),3)
    #     ens_mae = mean_absolute_error(y_true, y_pred_matrix['Ensamble'])
    #     ens_accuracy = accuracy_score(y_true, y_pred_matrix['Ensamble'])
    #
    #     temp_vet = ['Ensemble', round(ens_accuracy, 3), round(ens_mae, 3)]
    #     for i in range(len(ens_score)):
    #         temp_vet.append(ens_score[i])
    #     temp_vet.append(ens_score_ave)
    #
    #     score_data[str(score_data.shape[1])] = temp_vet
    #     return score_data

    def plot_roc_curve(self, y_prob_dict, y_true):
        """Plot the roc curve for base learners and ensemble.
        :param y_true: array with the targets, pandas (it is NOT onehot encoded)
        :param y_prob_dict: dictionary of pandas matrices with the probability resutls for all classes (it is produced by the prob_matrix_generator function)
        """
        labels = np.sort(y_true.unique())
        y_test_roc = self.onehotencoding_numerical_vector(y_true)
        # y_test_roc = label_binarize(y_true, classes=labels)

        name_mod = list(y_prob_dict.keys())
        n_classes = y_prob_dict[list(y_prob_dict.keys())[0]].shape[1]
        cm = [plt.cm.rainbow(i) for i in np.linspace(0, 1.0, n_classes + 1)]

        j = 0
        for name in y_prob_dict:
            ds().nuova_fig(j)
            ds().titoli(titolo= name_mod[j]+' prob', xtag='False Positive Rate', ytag='True Positive Rate')
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            i = 0
            for col in y_prob_dict[name].columns:
                fpr[i], tpr[i], _ = roc_curve(y_test_roc[:, i], y_prob_dict[name][col])
                roc_auc[i] = auc(fpr[i], tpr[i])
                ds().dati(fpr[i], tpr[i], colore=cm[i + 1], descrizione=str(labels[i])+' area '+str(round(roc_auc[i],3)))
                i = i + 1

            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            # Finally average it and compute AUC
            mean_tpr /= n_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            ds().dati(fpr['macro'], tpr['macro'], colore='black', descrizione=str('average')+' area '+str(round(roc_auc['macro'],3)))
            plt.plot([0, 1], [0, 1], 'k--')

            ds().legenda()
            ds().porta_a_finestra()
            j = j + 1

    def score_accuracy_recall(self, y_pred_matrix, y_true, verbose = 1):
        """evaluate accuracy recall and plot the confusion matrix for base learners and ensemble.
        :param y_true: array with the targets, pandas (it is NOT onehot encoded)
        :param y_pred_matrix: pandas matrix with the predicted class for each model as column, the prediction comes in the same format of the target (it is NOT onehot encoded)
        :param verbose: 0 or 1, with 1 the confusion matrix is ploted
        :return: accurary_real.shape[0] = y_pred_matrix.shape[0], accurary_real.shape[1] = (accuracy_tot, recall_average, precision_average), pandas matrix with the models and some scores
        """
        # labels = np.sort(y_true.unique())
        y_pred_matrix['Ensamble'] = y_pred_matrix['Ensamble'].map(round)
        accurary_real = pd.DataFrame()
        for col in y_pred_matrix.columns:
            cm = confusion_matrix(y_target=y_true, y_predicted=y_pred_matrix[col], binary=False)
            prob_cm =  np.zeros_like(cm).astype(float)
            # matrix with the predicted vs true values
            df_cm = pd.DataFrame(cm)

            # df_cm.columns = ['true '+str(i) for i in range(df_cm.shape[1])]
            # df_cm.index = ['predict '+str(i) for i in range(df_cm.shape[0])]

            tot_true_val = []
            tot_predic_val = []
            recall = []
            precision = []

            #calculate the recall and precision for each class separatly
            for i in range(cm.shape[0]):
                #create the initial matrix with the percentage of cases in each class
                #il 100% viene fatto sommando assieme tutti i casi in una colonna che rappresentano il totale dei casi reali
                prob_cm[i,:] = cm[i,:]/cm[i,:].sum()

                # la totalita dei casi reali per ciascuna classe viene calcolata sommando assieme tutti i casi in ciascuna colonna
                tot_true_val.append(cm[i,:].sum())# rapresents the TP (correct predicted data that are on the diagonal) + FN (tutti quelli che fanno parte di una classe ma che non tutti sono stati erroneamente messi qui)
                # la totalita dei casi predetti per ciascuna classe viene calcolata sommando assieme tutti i casi in ciascuna riga
                tot_predic_val.append(cm[:,i].sum())# rapresents the TP (correct predicted data that are on the diagonal) + FP (tutti quelli che sono statti predetti appartenere ad una classe anche se non tutti ne fanno parte)

                #calcolo del recupero (sensibilita) = TP/(TP+FN)
                if tot_true_val[i] == 0:
                    recall.append(0)
                else:
                    recall.append(round(cm[i,i]/tot_true_val[i],2))

                #calcolo della precisione = TP/(TP+FP)
                if tot_predic_val[i] == 0:
                    precision.append(0)
                else:
                    precision.append(round(cm[i,i]/tot_predic_val[i],2))

            #add to the main matrix the total true values for each classes
            df_cm['tot true'] = tot_true_val

            #add recall of each class to the matrix
            df_recall = pd.DataFrame(recall, columns = ['recall'])
            df_cm = pd.concat([df_cm, df_recall], axis = 1)

            #vector with the prediction of each class
            df_predict = pd.DataFrame(tot_predic_val, columns = ['tot predict']).T
            # calcola il numero totale di campioni sommando assieme tutti i campioni che realmente appartengono ad ciascuna classe
            tot_samples = df_cm['tot true'].sum()
            #aggiunge al vettore dei campioni che realmente appartengono a ciuscuna scalla il numero totale dei campioni
            df_predict['tot true'] = tot_samples
            #avegare recall from each class
            df_predict['recall'] = round(df_cm['recall'].sum()/len(recall),2)
            #add at the matrix the total prediction of each class, the total number of cases and the average recall
            df_cm = pd.concat([df_cm, df_predict])

            #create a matrix with the precision for each classes
            df_precision = pd.DataFrame(precision, columns = ['precision']).T
            #add to the matrix the average precision
            df_precision['tot true'] = round(df_precision.loc['precision'].sum()/len(precision),2)

            accuracy_tot = 0
            accuracy_average_tot = 0

            #viene calcolata l'accuratezza (TP+TN)/(TN+TP+FN+TP), modo complicato per dire tutto quello che e' stato predetto correttamente diviso il totale dei casi
            for i in range(cm.shape[0]):
                accuracy_tot = cm[i,i] + accuracy_tot# TP + TN che rappresentano tutti gli indovinati correttamente
                accuracy_average_tot = prob_cm[i,i]+accuracy_average_tot# TP + TN ma in precentuale
            accuracy_average_tot = accuracy_average_tot/len(cm[:,0])#TP+TN/nunmero_classi, rappresenta l'accuratezza media tra tutte le classi
            accuracy_tot = accuracy_tot/len(y_true)#somma di tutti i casi correttamente indovinati (TP+TN) diviso il numero di casi totali, rappresenta l'accuratezza totale

            #viene aggiunta al vettore recall il valore accuratezza totale
            df_precision['recall'] = round(accuracy_tot,2)
            #viene messo nella matrice principale il vettore con la precisione per ogni classe, la precisione media e l'accuratezza totale
            df_cm= pd.concat([df_cm, df_precision])

            temp_index = df_cm.index.tolist()[-2:]
            temp_index =  ['True '+str(i) for i in range(df_cm.shape[0]-2)] + temp_index
            df_cm.index = temp_index
            temp_col = df_cm.columns.tolist()[-2:]
            temp_col =  ['Predict '+str(i) for i in range(df_cm.shape[0]-2)] + temp_col
            df_cm.columns = temp_col

            accurary_real[col] = [accuracy_tot, accuracy_average_tot, df_cm['tot true']['precision'], df_cm['recall']['tot predict']]
            if verbose == 1:
                # fig, ax = plot_confusion_matrix(conf_mat=cm, show_normed=True, show_absolute=True, colorbar=True, class_names=labels)
                # plt.show()
                # sns.heatmap(df_cm, annot=True, annot_kws={'size': 8}, cmap=plt.cm.Blues, vmax=tot_samples, vmin=0, square=True, linewidths=0.5, linecolor="black")
                # sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, vmax=tot_samples, vmin=0, square=True, linewidths=0.5, linecolor="black", cbar=False)
                # plt.show()

                df_cm_ar = df_cm.to_numpy()
                df_cm_ar_temp = df_cm_ar.copy()
                fig, ax = plt.subplots()
                df_cm_ar_temp[-2:, -2:] = -500
                df_cm_ar_temp[:-2, -2:-1] = -200
                df_cm_ar_temp[-2:-1, :-2] = -200
                df_cm_ar_temp[:-2, -1] = -300
                df_cm_ar_temp[-1, :-2] = -300

                ax.imshow(df_cm_ar_temp, cmap=plt.get_cmap('cool'))
                ax.set_xticks(np.arange(df_cm_ar.shape[1]))
                ax.set_yticks(np.arange(df_cm_ar.shape[0]))
                ax.set_xticklabels(df_cm.columns.tolist())
                ax.set_yticklabels(df_cm.index.tolist())
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                for i in range(df_cm_ar.shape[0]):
                    for j in range(df_cm_ar.shape[1]):
                        ax.text(j, i, df_cm_ar[i, j], ha="center", va="center", color="black")

                ax.text(df_cm_ar.shape[1]-1, df_cm_ar.shape[0]-1-0.3, 'Ave Acc tot', ha="center", va="center", color="black")
                ax.text(df_cm_ar.shape[1]-1, df_cm_ar.shape[0]-2-0.3, 'Ave Recall', ha="center", va="center", color="black")
                ax.text(df_cm_ar.shape[1]-2, df_cm_ar.shape[0]-2-0.3, 'Tot Counts', ha="center", va="center", color="black")
                ax.text(df_cm_ar.shape[1]-2, df_cm_ar.shape[0]-1-0.3, 'Ave Prec', ha="center", va="center", color="black")

                ax.set_title(col)
                fig.tight_layout()
                plt.show()

        accurary_real = accurary_real.T
        accurary_real.columns = ['accuracy_tot', 'accuracy_average', 'precision_average', 'recall_average']
        return accurary_real


    def plot_precision_recall(self, y_prob_dict, y_true):
        """Plot the precision recall curve for base learners and ensemble.
        :param y_true: array with the targets, pandas (it is NOT onehot encoded)
        :param y_prob_dict: dictionary of pandas matrices with the probability resutls for all classes (it is produced by the prob_matrix_generator function)
        """
        labels = np.sort(y_true.unique())
        y_test_roc = self.onehotencoding_numerical_vector(y_true)
        # y_test_roc = label_binarize(y_true, classes=labels)

        name_mod = list(y_prob_dict.keys())
        n_classes = y_prob_dict[list(y_prob_dict.keys())[0]].shape[1]
        cm = [plt.cm.rainbow(i) for i in np.linspace(0, 1.0, n_classes + 1)]

        j = 0
        for name in y_prob_dict:
            ds().nuova_fig(j)
            ds().titoli(titolo= name_mod[j]+' prob', xtag='Recall', ytag='Precision')
            precision = dict()
            recall = dict()
            average_precision = dict()

            i = 0
            for col in y_prob_dict[name].columns:
                precision[i], recall[i], _ = precision_recall_curve(y_test_roc[:, i], y_prob_dict[name][col])
                average_precision[i] = average_precision_score(y_test_roc[:, i], y_prob_dict[name][col])
                ds().dati(recall[i], precision[i], colore=cm[i + 1], descrizione=str(labels[i])+' area '+str(round(average_precision[i],3)))
                i = i + 1
            ds().legenda()
            ds().porta_a_finestra()
            j = j + 1

    def plot_history(self, fitModel, type = 'dnn'):
        if fitModel != 0 and type == 'dnn':
            history_dict = fitModel.history
            history_dict.keys()
            loss = history_dict['loss']
            val_loss = history_dict['val_loss']

            acc = history_dict['acc']
            val_acc = history_dict['val_acc']
            epochs = range(1, len(loss) + 1)

            ds().nuova_fig(1, indice_subplot=211)
            ds().titoli(titolo="Training loss", xtag='Epochs', ytag='Loss', griglia=0)
            ds().dati(epochs, loss, descrizione = 'Training loss', colore='red')
            ds().dati(epochs, val_loss, descrizione = 'Validation loss')
            ds().dati(epochs, loss, colore='red', scat_plot ='scat', larghezza_riga =10)
            ds().dati(epochs, val_loss, scat_plot ='scat', larghezza_riga =10)
            ds().range_plot(bottomY =np.array(val_loss).mean()-np.array(val_loss).std()*6, topY = np.array(val_loss).mean()+np.array(val_loss).std()*6)
            ds().legenda()
            ds().nuova_fig(1,indice_subplot=212)
            ds().titoli(titolo="Training loss", xtag='Epochs', ytag='Accuracy', griglia=0)
            ds().dati(epochs, acc, descrizione = 'Training loss', colore='red')
            ds().dati(epochs, val_acc, descrizione = 'Validation loss')
            ds().dati(epochs, acc, colore='red', scat_plot ='scat', larghezza_riga =10)
            ds().dati(epochs, val_acc, scat_plot ='scat', larghezza_riga =10)
            ds().range_plot(bottomY =np.array(val_acc).mean()-np.array(val_acc).std()*6, topY = np.array(val_acc).mean()+np.array(val_acc).std()*6)
            ds().legenda()

            plt.show()
        elif fitModel !=0 and type == 'xgb':
            epochs = len(fitModel['validation_0']['error'])
            x_axis = range(0, epochs)
            # plot log loss
            fig, ax = plt.subplots()
            ax.plot(x_axis, fitModel['validation_0']['logloss'], label='Train')
            ax.plot(x_axis, fitModel['validation_1']['logloss'], label='Test')
            ax.legend()
            plt.ylabel('Log Loss')
            plt.title('XGBoost Log Loss')
            plt.show()
            # plot classification error
            fig, ax = plt.subplots()
            ax.plot(x_axis, fitModel['validation_0']['error'], label='Train')
            ax.plot(x_axis, fitModel['validation_1']['error'], label='Test')
            ax.legend()
            plt.ylabel('Classification Error')
            plt.title('XGBoost Classification Error')
            plt.show()
