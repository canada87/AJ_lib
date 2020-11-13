import sys
sys.path.insert(0, 'C:/Users/Max Power/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
sys.path.insert(0, 'C:/Users/ajacassi/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
from AJ_draw import disegna as ds
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import json

from tensorflow import keras

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import SGDRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.stats.stattools import durbin_watson

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA

from fbprophet import Prophet

class learning_reg:
    def __init__ (self, SEED = 222):
        self.SEED = SEED




    # ███    ███  ██████  ██████  ███████ ██      ███████
    # ████  ████ ██    ██ ██   ██ ██      ██      ██
    # ██ ████ ██ ██    ██ ██   ██ █████   ██      ███████
    # ██  ██  ██ ██    ██ ██   ██ ██      ██           ██
    # ██      ██  ██████  ██████  ███████ ███████ ███████




    def get_models(self, list_chosen):
        """Generate a library of base learners
        (Prophet works only if the data have the target in a pandas column named 'y' and a feature column with the tima data named 'ds')
        :param list_chosen: list with the names of the models to load
        :return: models, a dictionary with as index the name of the models, as elements the models"""

        linreg = LinearRegression(normalize=True, fit_intercept=True)
        dtr = DecisionTreeRegressor(random_state=self.SEED, min_samples_split=(0.018), min_samples_leaf= (0.007), max_depth=25)
        svrr = SVR(kernel='linear', epsilon=5)
        br = BaggingRegressor(n_estimators=350, max_samples=0.9, max_features=0.7, bootstrap=False, random_state=self.SEED)
        ada = AdaBoostRegressor(n_estimators=7, loss='exponential', learning_rate=0.01, random_state=self.SEED)
        rf = RandomForestRegressor(n_estimators=1000, max_depth= 30, max_leaf_nodes=1000, random_state=self.SEED)
        gbr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01,random_state=self.SEED)
        xgbr1 = xgb.XGBRegressor(random_state=self.SEED)
        mdl = LGBMRegressor(n_estimators=1000, learning_rate=0.01)
        las = Lasso()
        rid = Ridge()
        en = ElasticNet()
        huber = HuberRegressor(max_iter=2000)
        lasl = LassoLars(max_iter=2000, eps = 1, alpha=0.5, normalize=False)
        pa = PassiveAggressiveRegressor(C=1, max_iter=4000, random_state=self.SEED)
        sgd = SGDRegressor(max_iter=2000, tol=1e-3)
        knn = KNeighborsRegressor(n_neighbors=20)
        ex = ExtraTreeRegressor()
        exs = ExtraTreesRegressor(n_estimators=1000)
        pro = Prophet(changepoint_prior_scale = 0.01)

        models_temp = {
            'BaggingRegressor': br,
            'RandomForestRegressor': rf,
            'GradientBoostingRegressor': gbr,
            'XGBRegressor': xgbr1,
            'LGBMRegressor':mdl,
            'ExtraTreesRegressor': exs,
            'LinearRegression': linreg,
            'SVR': svrr,
            'AdaBoostRegressor': ada,
            'LassoLars': lasl,
            'PassiveAggressiveRegressor': pa,
            'SGDRegressor': sgd,
            'DecisionTreeRegressor': dtr,
            'lasso': las,
            'ridge': rid,
            'ElasticNet': en,
            'HuberRegressor': huber,
            'KNeighborsRegressor': knn,
            'ExtraTreeRegressor': ex,
            'Prophet': pro}

        models = dict()
        for model in list_chosen:
            if model in models_temp:
                models[model] = models_temp[model]
        return models

    def get_ARIMA_models(self, y, order, x = None, type = 'ARIMA'):
        '''
        :param y: array with the targets, pandas series
        :param order: (p,d,q)
        :param x: matrix with the features, pandas. Used only if exogenous variables are present
        :param type: ARIMA, SARIMAX
        :return: model, ARIMA model
        '''
        if type == 'ARIMA':
            model = ARIMA(y, order = order, exog=x)
        elif type == 'SARIMAX':
            model = SARIMAX(y, order = (order[0], order[1], order[2]), seasonal_order=(order[3], order[4], order[5], order[6]), exog=x)
        return model

    def get_VAR_models(self, data, exog_data = None, order = None, type = 'VAR'):
        '''
        generate the model VAR. Vector Autoregression (VAR) is a multivariate forecasting algorithm that is used when two or more time series influence each other.
        You need atleast two time series (variables). The time series should influence each other.
        :param data: matrix with the all data, pandas. The model will try to predict the next value for each of the features.
        :param exog_train: If some features are non strictly influenced can be put in this matrix, pandas
        :param order: (p,q) order of the model for the number of AR and MA parameters to use, needed only with VARMAX
        :param type: VAR, VARMAX
        :return: model
        '''
        if type == 'VAR':
            model = VAR(data, exog=exog_data)
        if type == 'VARMAX':
            model = VARMAX(data, exog=exog_data, order=order)
        return model

    def get_deep_learning_model(self, input_dl, output_dl = 1, active = 'linear', lost = 'mae', net_type = 'normal'):
        ''' generate the deep learning model'''
        # model = keras.Sequential()
        # act = 'relu'
        # model.add(keras.layers.Dense(4, activation=act, input_dim = input_dl))
        # model.add(keras.layers.Dense(1, activation = active))
        # opt = keras.optimizers.SGD(lr=0.04,momentum=0.9)
        # model.compile(optimizer = opt, loss = lost, metrics=['mae'])
        # return model

        if net_type == 'normal':
            model = keras.Sequential()
            model.add(keras.layers.Dense(256, input_dim=input_dl, activation='relu'))
            model.add(keras.layers.Dense(256, activation='relu'))
            model.add(keras.layers.Dense(128, activation='relu'))
            model.add(keras.layers.Dense(64, activation='relu'))
            model.add(keras.layers.Dense(32, activation='relu'))
            model.add(keras.layers.Dense(output_dl, activation='relu'))
            model.compile(loss=lost, optimizer = 'adam', metrics = ['accuracy'])

        if net_type == 'vgg':
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
            output = keras.layers.Dense(output_dl, activation = 'relu')(hidden1)
            model = keras.models.Model(inputs = visible, outputs = output)
            model.compile(optimizer = 'adam', loss = lost, metrics=['accuracy'])
        return model



    # ████████ ██████   █████  ██ ███    ██ ██ ███    ██  ██████
    #    ██    ██   ██ ██   ██ ██ ████   ██ ██ ████   ██ ██
    #    ██    ██████  ███████ ██ ██ ██  ██ ██ ██ ██  ██ ██   ███
    #    ██    ██   ██ ██   ██ ██ ██  ██ ██ ██ ██  ██ ██ ██    ██
    #    ██    ██   ██ ██   ██ ██ ██   ████ ██ ██   ████  ██████




    def train_models(self, models, xtrain, ytrain, epochs = 0, validation_data = None, shuffle = True, batch_size = 32):
        '''training function
        (Prophet works only if the data have the target in a pandas column named 'y' and a feature column with the tima data named 'ds')
        :param models: dictionary with as index the name of the models, as elements the models
        :param xtrain: matrix with the features, pandas or numpy
        :param ytrain: array with the targets, pandas or numpy
        :param epochs: epochs used if deep learning is in the models, or VAR as lag
        :param validation_data: data used to validate if deep learning is used, pandas or numpy
        :param shuffle: Boolean, used to shuffle the data before the training if deep learning is used
        :return: models, a dictionary with as index the name of the models, as elements the models after the training'''
        fitModel = 0
        for i, (name_model, model) in enumerate(models.items()):
            if name_model == 'deep learning normal':
                fitModel = model.fit(xtrain, ytrain, epochs = epochs, verbose = 1, validation_data= validation_data, shuffle = shuffle, batch_size = batch_size)
            elif name_model == 'deep learning vgg':
                xvgg = xtrain.copy()
                xvgg = xvgg.to_numpy()
                xvgg = xvgg.reshape(xvgg.shape[0], xvgg.shape[1], 1)
                validation_data_new = None
                if validation_data:
                    xvgg_val = validation_data[0]
                    xvgg_val = xvgg_val.to_numpy()
                    xvgg_val = xvgg_val.reshape(xvgg_val.shape[0], xvgg_val.shape[1], 1)
                    validation_data_new = (xvgg_val,validation_data[1])
                fitModel = model.fit(xvgg, ytrain, epochs = epochs, verbose = 1, validation_data= validation_data_new, shuffle = shuffle, batch_size = batch_size)
            elif name_model == 'ARIMA':
                model.fit(trend = 'nc', disp = 0)
            elif name_model == 'VAR':
                model.fit(epochs)# epochs represent the lag
            elif name_model == 'VARMAX':
                model.fit()
            elif name_model == 'Prophet':
                for col in xtrain.columns:
                    if col != 'y' and col != 'ds':
                        model.add_regressor(col)
                model.fit(xtrain)
            else:
                model.fit(xtrain, ytrain)
        return models, fitModel

    def train_hyperparameters(self, models, xtrain, ytrain, dict_grid, filename = 'param_file'):
        '''Training function for the hyperparameters search, valid only for the standard models
        :param models: dictionary with as index the name of the models, as elements the models
        :param xtrain: matrix with the features, pandas or numpy
        :param ytrain: array with the targets, pandas or numpy
        :param dict_grid: dictionary with the list of the paramters and the limits for the search
        :param filename: str, file name of the list with the resutls
        :return: dict_best_param, dictionary with the best parameters for each model in the models
        '''
        dict_best_param = {}
        for i, (name_model, model) in enumerate(models.items()):
            model_random = GridSearchCV(estimator=model, param_grid=dict_grid[name_model], cv=3, verbose=2, n_jobs=-1)
            model_random.fit(xtrain, ytrain)
            dic_h = model_random.best_params_
            dict_best_param[name_model] = dic_h
        with open(filename, 'w') as f:
            json.dump(dict_best_param, f)
        return dict_best_param

    def train_horizontal_ensamble(self, models, xtrain, ytrain, epochs, horizontal_perc = 0.1, shuffle = True):
        '''horizontal ensamble, valid only for the deep learning model
        :param models: dictionary with as index the name of the models, as elements the models
        :param xtrain: matrix with the features, pandas or numpy
        :param ytrain: array with the targets, pandas or numpy
        :param epochs: epochs used if deep learning is in the models
        :param horizontal_perc: from 0 to 1, percentual of epochs used to calculate the ensample
        :param shuffle: Boolean, used to shuffle the data before the training if deep learning is used
        :return: models, a dictionary with as index the name of the models, as elements the models where the deep learning is a list of models
        '''

        def load_horizontal_ensamble(model_list, epochs):
            all_models = list()
            for epoc in range(int(epochs - epochs*0.1), epochs):
                sing_model = keras.models.load_model('modelli/model_'+str(epoc)+'.h5')
                all_models.append(sing_model)
            model_list['deep learning'] = all_models
            return model_list

        name_model = 'deep learning'
        model = models[name_model]

        for epo in range(epochs):
            model.fit(xtrain, ytrain, epochs = 1, verbose = 1, shuffle = shuffle)
            if epo>=int(epochs - epochs*horizontal_perc):
                model.save('modelli/model_'+str(epo)+'.h5')

        models = load_horizontal_ensamble(models, epochs)
        return models

    def train_rolling_window(self, models, X_tot, Y_tot, frame_perc, epochs = 1, validation = False):
        '''train with rolling window, the function can be expanded to use the validation for checking purposes.
            This is not a real rolling, it train the dataset a frame at a time, but it doesn't use the window to move
            across time, it works if time is along the column like in the case of time and other features. If you have time only this isn't the
            right one.
        :param models: dictionary with as index the name of the models, as elements the models
        :param X_tot: matrix with the features, pandas or numpy
        :param Y_tot: array with the targets, pandas or numpy
        :param frame_perc: from 0 to 1, percentual of the dataset which create the size of the rolling window
        :param epochs: epochs used if deep learning is in the models
        :param validation: boolean, with True 30% of the data are use to validate if using deep learning model
        '''
        frame_size = int(X_tot.shape[0]*frame_perc)
        last_step = round(X_tot.shape[0]/frame_size)*frame_size - X_tot.shape[0]

        for step in range(round(X_tot.shape[0]/frame_size)):
            start_frame = step*frame_size
            end_frame = start_frame + frame_size
            x_frame = X_tot[start_frame:end_frame]
            y_frame = Y_tot[start_frame:end_frame]
            if validation:
                x_train, x_val, y_train, y_val = train_test_split(x_frame, y_frame, shuffle = False, test_size = 0.3)
                xy_val = (x_val, y_val)
            else:
                x_train, y_train = x_frame, y_frame
                xy_val = None
            models, _ = self.train_models(models, x_train, y_train, epochs = epochs, validation_data = xy_val, shuffle = False)

        if last_step != 0:
            x_frame = X_tot[last_step:]
            y_frame = Y_tot[last_step:]
            models, _ = self.train_models(models, x_frame, y_frame, epochs = epochs, validation_data = None, shuffle = False)
        return models

    def train_rolling_window2(self, models, x_train, y_train, rol_frame, epochs = 1):
        '''train with rolling window. This is the real rolling. It takes as imput the time line and gives as autput a time line. It rolls on the time line a window with
        a rol_frame size and it include the data used as Y to the X while it is rolling.
        :param models: dictionary with as index the name of the models, as elements the models
        :param x_train: matrix of time lines, pandas or numpy, the time line has to lie on the shape[1]
        :param y_train: matrix of time lines representing the next section of the x_train, pandas or numpy, the time line has to lie on the shape[1]
        :param rol_frame: int, size of the window to roll
        :param epochs: epochs used if deep learning is in the models
        '''

        print()
        frame_size = x_train.shape[1]
        print('X frame size', frame_size)
        test_frame = y_train.shape[1]
        print('Y frame size', test_frame)
        print()

        step_size = int(test_frame/rol_frame)
        last_step = test_frame - step_size*rol_frame

        for step in range(step_size):
            start_frame = step*rol_frame
            end_frame = start_frame + frame_size
            x_frame = x_train.iloc[:,start_frame:end_frame]
            y_frame = y_train.iloc[:,step*rol_frame:step*rol_frame + rol_frame]
            models, _ = self.train_models(models, x_frame, y_frame, epochs = epochs)
            x_train = pd.concat([x_train.iloc[:,:], y_frame], axis = 1)

        if last_step != 0:
            x_frame = x_train.iloc[:,-frame_size:]
            y_frame = y_train.iloc[:,-rol_frame:]
            models, _ = self.train_models(models, x_frame, y_frame, epochs = epochs)
        return models

    def train_negative_binomial_model(self, x_train, y_train, test_size):
        '''generate and train the negative binomial model
        :param xtrain: matrix with the features, pandas or numpy
        :param ytrain: array with the targets, pandas or numpy
        :param test_size: from 0 to 1, percentual of train to use as test in the parameter evaluatio of the model
        :return: negative binomial model fitted
        '''
        x_train_train, x_train_test, y_train_train, y_train_test = train_test_split(x_train, y_train, shuffle = False, test_size = test_size)
        train = x_train_train.copy()
        test = x_train_test.copy()
        train['target'] = y_train_train
        test['target'] = y_train_test

        # Step 1: specify the form of the model
        model_formula = train.columns[0]
        for i in range(1, len(train.columns)-1):
            model_formula = model_formula+" + "+train.columns[i]
        model_formula = train.columns[-1] + ' ~ ' + model_formula

        grid = 10 ** np.arange(-8, -3, dtype=np.float64)
        best_alpha = []
        best_score = 1000
        # Step 2: Find the best hyper parameter, alpha
        for alpha in grid:
            model = smf.glm(formula=model_formula, data=train, family=sm.families.NegativeBinomial(alpha=alpha))
            results = model.fit()
            predictions = results.predict(test).astype(int)
            score = eval_measures.meanabs(predictions, test.total_cases)
            if score < best_score:
                best_alpha = alpha
                best_score = score

        # fit the final model
        data = x_train.copy()
        data['target'] = y_train
        model_formula = data.columns[0]
        for i in range(1, len(data.columns)-1):
            model_formula = model_formula+" + "+data.columns[i]
        model_formula = data.columns[-1] + ' ~ ' + model_formula

        # # Step 4: refit on entire dataset
        model = smf.glm(formula=model_formula, data=data, family=sm.families.NegativeBinomial(alpha=best_alpha))
        fitted_model = model.fit()

        return fitted_model




    # ███████  ██████  ██████  ███████  ██████  █████  ███████ ████████
    # ██      ██    ██ ██   ██ ██      ██      ██   ██ ██         ██
    # █████   ██    ██ ██████  █████   ██      ███████ ███████    ██
    # ██      ██    ██ ██   ██ ██      ██      ██   ██      ██    ██
    # ██       ██████  ██   ██ ███████  ██████ ██   ██ ███████    ██




    def predict_matrix_generator(self, models, xtest):
        '''generate the prediction with all the model in the list and add all of them to the same matrix, adding the average prediciton (ensamble)
        :param models: dictionary with as index the name of the models, as elements the models
        :param xtest: matrix with the features, pandas or numpy
        :return: Predict_matrix.shape[0] = xtest.shape[0], Predict_matrix.shape[1] = len(models) + 1, Pandas matrix with the predicion for all the models and the average for the ensamble
        '''
        Predict_matrix = pd.DataFrame()
        cols = list()
        for i, (name_model, model) in enumerate(models.items()):
            if name_model == 'SARIMAX' or name_model == 'ARIMA':
                Predict_matrix[name_model] = model.forecast(xtest.shape[0], exog = xtest).reset_index(drop = True)
            elif name_model == 'AUTOARIMA':
                Predict_matrix[name_model], confint = model.predict(n_periods=xtest.shape[0], return_conf_int=True, exogenous=xtest)
            elif name_model == 'VAR':
                self.forecast_priority = None # rappresenta i valori del training su cui verra proritizzata la predizione per il futuro, di solito sono le ultime entries prima del test_set, pandas
                Predict_matrix[name_model] = model.forecast(steps = xtest.shape[0], y = self.forecast_priority)
            elif name_model == 'VARMAX':
                self.exog_test = None # If some features are non strictly influenced can be put in this matrix, pandas
                Predict_matrix[name_model] = model.forecast(steps = xtest.shape[0], exog = self.exog_test)
            elif name_model == 'deep learning normal':
                Predict_matrix[name_model] = model.predict(xtest)[:,0]
            elif name_model == 'deep learning vgg':
                xvgg = xtest.copy()
                xvgg = xvgg.to_numpy()
                xvgg = xvgg.reshape(xvgg.shape[0], xvgg.shape[1], 1)
                Predict_matrix[name_model] = model.predict(xtest)[:,0]
            elif name_model == 'NegativeBinomial':
                Predict_matrix[name_model] = model.predict(xtest).to_numpy()
            elif name_model == 'Prophet':
                future = pd.DataFrame(xtest['ds'])
                Predict_matrix[name_model] = model.predict(future)['yhat']
            else:
                Predict_matrix[name_model] = model.predict(xtest)
            cols.append(name_model)
        Predict_matrix['Ensamble'] = Predict_matrix.mean(axis=1)
        return Predict_matrix

    def predict_rolling_ARIMA(self, ytrain, ytest, order, type):
        '''train ARIMA with rolling, it can be used to forecast as well
        :param ytrain: array with the targets, pandas series
        :param ytest: array with the targets, pandas series. If used to forecast the ytest has to be an empty series with the len of the prediction and the index of the temporal range
        :param order: (p,d,q)
        :param type: ARIMA, SARIMAX
        :return: ypred.shape[0] = ytest.shape[0], pandas series with the index of ytest
        '''
        history = [x for x in ytrain]
        ypred = list()
        models = dict()
        for t in range(ytest.shape[0]):
            models[type] = self.get_ARIMA_model(history, order = order, type=type)
            models[type] = self.train_models(models, 0, 0)
            pred = models[type].forecast()[0]
            ypred.append(pred)
            history.append(pred)
        ypred = pd.Series(ypred)
        ypred.index = ytest.index
        return ypred

    def predict_rolling_VAR(self, train, test, lag=0, order = None, type = 'VAR'):
        '''train VAR with rolling, it can be used to forecast as well.
        :param train: matrix with the all data to train, pandas. The model will try to predict the next value for each of the features.
        :param test: matrix with the all data to test and predict, pandas. The model will try to predict the next value for each of the features.
        :param lag: lag in the training for VAR model
        :param order: (p,q) order of the model for the number of AR and MA parameters to use, needed only with VARMAX
        :param type: VAR, VARMAX
        :return: final_pred.shape[0] = test.shape[0], final_pred.shape[1] = train.shape[1], pandas matrix
        '''
        history = train.copy()
        final_pred = pd.DataFrame()
        models = dict()
        for t in range(test.shape[0]):
            models[type] = self.get_VAR_models(history, order = order, type = type)
            models[type] = self.train_models(models, 0, 0, epochs=lag)
            pred = models[type].forecast(steps = 1)
            pred = pd.DataFrame(pred, columns = train.cloumns)
            final_pred = pd.concat([final_pred, pred], ignore_index = True)
            history = pd.concat([history, pred], ignore_index = True)
        return final_pred

    def predict_rolling_window2(self, models, x_test, predict_frame, rol_frame):
        print()
        frame_size = x_test.shape[1]
        print('X frame size', frame_size)
        print()

        x_test.columns = [i for i in range(len(x_test.columns.tolist()))]

        step_size = int(predict_frame/rol_frame)
        last_step = predict_frame - step_size*rol_frame

        for i, (name_model, model) in enumerate(models.items()):

            for step in range(step_size):
                start_frame = step*rol_frame
                end_frame = start_frame + frame_size
                x_frame = x_test.iloc[:,start_frame:end_frame]
                predict_matrix = model.predict(x_frame)
                predict_matrix = pd.DataFrame(predict_matrix)
                predict_matrix.index = x_frame.index
                predict_matrix.columns = [i+x_frame.columns.tolist()[-1]+1 for i in range(rol_frame)]
                x_test = pd.concat([x_test.iloc[:,:], predict_matrix], axis = 1)

            if last_step != 0:
                x_frame = x_test.iloc[:,-frame_size:]
                predict_matrix = model.predict(x_frame)
                predict_matrix = pd.DataFrame(predict_matrix)
                predict_matrix.index = x_frame.index
                predict_matrix.columns = [i+x_frame.columns.tolist()[-1]+1 for i in range(rol_frame)]
                x_test = pd.concat([x_test.iloc[:,:], predict_matrix], axis = 1)

        return x_test.iloc[:,-predict_frame:]

        # ███████  ██████  ██████  ██████  ███████
        # ██      ██      ██    ██ ██   ██ ██
        # ███████ ██      ██    ██ ██████  █████
        #      ██ ██      ██    ██ ██   ██ ██
        # ███████  ██████  ██████  ██   ██ ███████



    def score_VAR_correlation(self, models, x_train, lag = 0, maxlag=None):
        '''
        durbin_watson test
        the closer the result is to 2 then there is no correlation, the closer to 0 or 4 then correlation implies
        '''
        for i, (name_model, model) in enumerate(models.items()):
            if name_model == 'VAR':
                if maxlag != None:#studio hypersapce sul parametro lag
                    vet_aic = []
                    vet_bic = []
                    vet_fpe = []
                    vet_hqic = []
                    for i in range(maxlag):
                        result = model.fit(i)
                        vet_aic.append(result.aic)
                        vet_bic.append(result.bic)
                        vet_fpe.append(result.fpe)
                        vet_hqic.append(result.hqic)
                    df_results = pd.DataFrame()
                    df_results['AIC'] = vet_aic
                    df_results['BIC'] = vet_bic
                    df_results['FPE'] = vet_fpe
                    df_results['HQIC'] = vet_hqic
                    return df_results
                else:# fit diretto su un valore specifico di lag
                    result = model.fit(lag)
                    out = durbin_watson(result.resid)
                    df_results = pd.DataFrame()
                    for col, val in zip(x_train.columns, out):
                        df_results[col] = [round(val, 2)]
                    return df_results.T

            elif name_model == 'VARMAX':
                result = model.fit()
                out = durbin_watson(result.resid)
                df_results = pd.DataFrame()
                for col, val in zip(x_train.columns, out):
                    df_results[col] = [round(val, 2)]
                return df_results.T

    def score_models(self, y_test, Predict_matrix):
        '''
        generate the score with the actual test target
        :param y_test: array with the actual targets, pandas or numpy
        :param Predict_matrix: array with the predicted targets for each model, pandas
        :return: df_score.shape[0] = num of models + 1, df_score.shape[1] = 5 (MAE, MSE, R2, R2 a mano, residual)
        '''
        df_score = pd.DataFrame()
        for name_model in Predict_matrix:
            SS_residual = ((y_test - Predict_matrix[name_model])**2).sum()
            df_y = pd.DataFrame()
            df_y['y_test'] = y_test
            df_y['y_pred'] = Predict_matrix[name_model]
            df_y['diff'] = (y_test - Predict_matrix[name_model])

            SS_Total = ((y_test - np.mean(y_test))**2).sum()
            r_square = 1 - (float(SS_residual))/SS_Total

            mae = mean_absolute_error(y_test, Predict_matrix[name_model])
            mse = mean_squared_error(y_test, Predict_matrix[name_model])
            r2 = r2_score(y_test, Predict_matrix[name_model])
            df_score[name_model] = [round(mae,3), round(mse,3), round(r2,3), round(r_square,3), round(SS_residual,3)]

        df_score = df_score.T
        df_score.columns = ['MAE', 'MSE', 'R2', 'R2 a mano', 'residual']
        return df_score


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
