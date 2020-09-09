import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.utils import resample
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, accuracy_score

from xgboost import XGBRegressor
import pickle
from lib_models_classifier import learning_class

class peeking:
    def __init__(self, data = 0):
        self.data = data

    def info(self):
        print('HEAD')
        print(self.data.head())
        print('')
        print('DESCRIBE')
        print(self.data.describe())
        print('COLUMNS')
        print(self.data.columns)
        print('')
        print('DATA INFO')
        print(self.data.info())
        print('')
        print('NAN PRESENCE')
        print(self.data.isna().sum())
        print('UNIQUE VALUE')
        for col in self.data.columns:
            print(col, self.data[col].value_counts().count())
            # print(col, self.data[col].nunique())

    def plot_mean(self, col_x, col_y):
        y = self.data.groupby(col_x)[col_y].mean()
        x = self.data[col_x].unique()
        plt.scatter(x, y)
        plt.ylabel(col_y)
        plt.xlabel(col_x)
        plt.show()

    def plot_correlation_roy(self, col_y):
        for i in range(0, len(self.data.columns), 5):
            sns.pairplot(data=self.data, x_vars = self.data.columns[i:i+5], y_vars = [col_y])
        plt.show()

class learning:
    def __init__(self, data = 0):
        self.data = data

        # ██████   █████  ████████  █████      ███    ███  █████  ███    ██ ██ ██████  ██    ██ ██       █████  ████████ ██  ██████  ███    ██
        # ██   ██ ██   ██    ██    ██   ██     ████  ████ ██   ██ ████   ██ ██ ██   ██ ██    ██ ██      ██   ██    ██    ██ ██    ██ ████   ██
        # ██   ██ ███████    ██    ███████     ██ ████ ██ ███████ ██ ██  ██ ██ ██████  ██    ██ ██      ███████    ██    ██ ██    ██ ██ ██  ██
        # ██   ██ ██   ██    ██    ██   ██     ██  ██  ██ ██   ██ ██  ██ ██ ██ ██      ██    ██ ██      ██   ██    ██    ██ ██    ██ ██  ██ ██
        # ██████  ██   ██    ██    ██   ██     ██      ██ ██   ██ ██   ████ ██ ██       ██████  ███████ ██   ██    ██    ██  ██████  ██   ████

    def merge_and_remove(self, vet1, vet2, element):
        vet2.append(element)
        vect = np.unique(vet1 + vet2)
        list_vect = vect.tolist()
        list_vect.remove(element)
        return list_vect

    def data_subset(self, frazione = 0.1):
        subset_data = self.data.sample(frac = frazione, random_state = 1)
        return subset_data

    def convertIntoNumbers(self, df, columns):
        df[columns]=LabelEncoder().fit_transform(df[columns])
        # in forma piu esplicita
        # label_encoder = LabelEncoder()
        # label_encoder = label_encoder.fit(df[columns])
        # df[columns] = label_encoder.transform(df[columns])
        # dict_class = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
        return df

    def createDummy(self, column):
        matchTypeDummies = pd.get_dummies(self.data[column],drop_first=True)
        data_w_dummies = pd.concat([self.data,matchTypeDummies],axis=1)
        return data_w_dummies

    def onehotencoding_numerical_vector(self, vector, n_values):
        '''
        take a vector of valuse [1,0,2] and gives the one hot encoding of that vector
        works only with numbers
        n_values is the number of values in the vector, es -> [0,4,2] = 5 ... because the range is from 0 to 4
        '''

        onehot_vector = np.zeros([len(vector), n_values])
        onehot_vector[np.arange(len(vector)), vector] = 1
        return onehot_vector

    def removingObj(self, obj = [np.inf, -np.inf]):
        clean_data = self.data.replace(obj, np.nan)
        return clean_data.dropna(axis=0)

    def dropCol(self, columns):
        droped_data = self.data.drop(columns,axis=1)
        return droped_data

    def dropRow(self, column, row, columns):
        """
        tutti gli elementi 'row' presenti in 'column' vengono salvati, vengono droppate le 'columns'
        """
        dropped_data = self.data.loc[self.data[column] == row].drop(columns,axis=1)
        return dropped_data

    def test_split(self, col_y, test_size = 0.3):
        x = self.data.drop([col_y],axis=1)
        y = self.data[[col_y]]
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=test_size)
        return X_train, X_test, y_train, y_test, x, y

    def bootstrapping_oob_sampling(self, df, n_samples):
        '''
        take a single pandas dataframe and the number of data in the boot (Out Of the Bag)
        it create 2 new dataframe, one with the number of data selected and the other with the rest
        according with the bootstrapping, in the boot dataframe is allowed the multiselection, this mean that the same data can be present multiple time
        '''
        df_ind = df.index.tolist()
        sample = resample(df, n_samples=n_samples)
        sample_ind = sample.index.tolist()
        oob_ind = np.array([x for x in df_ind if x not in sample_ind])
        oob = df.loc[oob_ind].copy()
        return sample, oob

    def kfold_sampling(df, n_samples, df_y = None, shuffle = False, type = 'kfold'):
        '''
        different type of kfolding -> kfold (standard one), stratified (every fold has the same amount of element of each class). timeseries (generate a moving forward dataset)
        :param df: pandas matrix
        :param n_samples: int, number of fold to split the data
        :param df_y: can be multiple things, pandas Series (or pandas single column) with the targets of a class problem, or int as the maximum number of data present in the each fold for the time series split
        :param shuffle: boolean, generate a ramdomize order of data (it doesn't work with the timeseries)
        :param type: 'kfold', 'stratified', 'timeseries'
        :return: train_dict, test_dict, are two dictionary where each element if a pandas matrix with the same number of columns of df but with a subset of rows
        '''
        vetor_index = np.arange(df.shape[0])

        if type == 'kfold':
            kfold = KFold(n_samples, shuffle)
            unfolder = kfold.split(vetor_index)
        elif type == 'stratified':
            kfold = StratifiedKFold(n_samples, shuffle)
            y = df_y.to_numpy()
            unfolder = kfold.split(vetor_index, y)
        elif type == 'timeseries':
            kfold = TimeSeriesSplit(n_samples, max_train_size = df_y)
            unfolder = kfold.split(vetor_index)

        train_dict = dict()
        test_dict = dict()
        for i, (train, test) in enumerate(unfolder):
            df_train = df.iloc[train].copy()
            df_test = df.iloc[test].copy()
            if shuffle and type != 'timeseries':
                train_dict[i] = df_train.sample(frac = 1)
                test_dict[i] = df_test.sample(frac = 1)
            else:
                train_dict[i] = df_train
                test_dict[i] = df_test
        return train_dict, test_dict


        # ███████ ███████  █████  ████████ ██    ██ ██████  ███████     ███████ ███████ ██      ███████  ██████ ████████ ██  ██████  ███    ██
        # ██      ██      ██   ██    ██    ██    ██ ██   ██ ██          ██      ██      ██      ██      ██         ██    ██ ██    ██ ████   ██
        # █████   █████   ███████    ██    ██    ██ ██████  █████       ███████ █████   ██      █████   ██         ██    ██ ██    ██ ██ ██  ██
        # ██      ██      ██   ██    ██    ██    ██ ██   ██ ██               ██ ██      ██      ██      ██         ██    ██ ██    ██ ██  ██ ██
        # ██      ███████ ██   ██    ██     ██████  ██   ██ ███████     ███████ ███████ ███████ ███████  ██████    ██    ██  ██████  ██   ████

    def Principal_Component_Analysis(self, x_train, x_test, num_componenti = None, plot_histo = 'yes'):
        pca = PCA(n_components=num_componenti)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)
        explained_varianze = pca.explained_variance_ratio_

        if plot_histo == 'yes':
            plt.bar(range(1,len(explained_varianze)+1), explained_varianze, alpha=0.5, align='center', label='individual explained variance')
            plt.ylabel('Explained variance ratio')
            plt.xlabel('Principal components')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.show()

        return x_train, x_test

    def correlation_matrix(self, col_y, corr_value = 0.95, corr_value_w_targhet = 0.95, plot_matr = 'yes'):
        """
        'col_y' represent the target column
        """
        corr = self.data.corr().abs()

        if plot_matr == 'yes':
            sns.heatmap(corr[(corr >= 0.5)], cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1, annot=True, annot_kws={"size": 8}, square=True, linecolor="black");
            plt.show()

        corr_with_target = corr[col_y]#correlation with the target
        relevant_feature_with_target = corr_with_target[corr_with_target < corr_value_w_targhet]

        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))#select upper triangle of correlation matrix
        correlation_between_parameters = [column for column in upper.columns if any(upper[column] > corr_value)]
        return relevant_feature_with_target.index, correlation_between_parameters

    def feature_importance(self, x_train, file_name = 'model.sav'):
        model = pickle.load(open(file_name, 'rb'))
        feature_importance = model.feature_importances_
        feature_df = pd.DataFrame(data=feature_importance, index=x_train.columns, columns=['importance'])
        feature_df.sort_values(by='importance', inplace=True)
        return feature_df

    def feature_selection_with_model(self, models, X_train, Y_train, X_test, Y_test = [], dict_index = dict(), thresh_results = 0, num_classes = 2, verbose = 0):
        '''
        usa un modello gia fittato per stabilire l'importanza delle feature e poi rifitta eliminando tutte le features partendo da quelle meno importanti
        dando come risultato l'accuratezza in funzione delle feature presenti
        '''

        learn_new = learning_class()
        model_list = []
        for model in models:
            model_list.append(model)
        models_new = learn_new.get_models(model_list)

        if len(Y_test) > 0:
            thresh_results = pd.DataFrame()
            acc_results = pd.DataFrame()
            num_results = pd.DataFrame()

            for model in models:
                feature_importance = models[model].feature_importances_
                feature_df = pd.DataFrame(data=feature_importance, index=X_train.columns, columns=['importance'])
                feature_df.sort_values(by='importance', inplace=True)
                if verbose == 1:
                    plt.bar(feature_df.index.tolist(), feature_df['importance'].tolist())
                    plt.xticks(rotation=90)
                    plt.show()

                thresholds = feature_df['importance'].to_numpy()
                num_param = []
                acc_vet = []
                for thresh in thresholds:
                    selection = SelectFromModel(models[model], threshold=thresh, prefit=True)
                    select_X_train = selection.transform(X_train)
                    selection_model = models_new[model]
                    selection_model.fit(select_X_train, Y_train)
                    select_X_test = selection.transform(X_test)
                    predictions = selection_model.predict(select_X_test)
                    accuracy = accuracy_score(Y_test, predictions)
                    num_param.append(select_X_train.shape[1])
                    acc_vet.append(accuracy)
                thresh_results[model] = thresholds
                num_results[model] = num_param
                acc_results[model] = acc_vet
            return thresh_results, num_results, acc_results

        else:

            y_pred_matrix = pd.DataFrame()
            y_prob_dict = {}
            for model in models:
                thresh = thresh_results[model].loc[dict_index[model]]
                selection = SelectFromModel(models[model], threshold=thresh, prefit=True)
                select_X_train = selection.transform(X_train)
                selection_model = models_new[model]
                selection_model.fit(select_X_train, Y_train)
                select_X_test = selection.transform(X_test)
                y_pred_matrix[model] = selection_model.predict(select_X_test)
                y_prob_dict[model] = pd.DataFrame(selection_model.predict_proba(select_X_test))

            Prob_ens = np.zeros((select_X_test.shape[0], num_classes))
            Prob_ens = pd.DataFrame(Prob_ens)
            for name in y_prob_dict:
                Prob_ens = y_prob_dict[name] + Prob_ens
            Prob_ens = Prob_ens/len(y_prob_dict)
            y_prob_dict['Ensamble'] = Prob_ens
            y_pred_matrix['Ensamble'] = round(y_pred_matrix.mean(axis=1))
            return y_prob_dict, y_pred_matrix

    def Recursive_Feature_Elimination(self, X_train, X_test, y_train, y_test, x, y, file_name = 'model.sav'):
        nof_list = np.arange(1, len(x.columns))
        high_score=0
        nof=0
        score_list =[]
        for n in range(len(nof_list)):
            model = LinearRegression()
            rfe = RFE(model, nof_list[n])
            X_train_rfe = rfe.fit_transform(X_train, y_train)
            X_test_rfe = rfe.transform(X_test)
            model.fit(X_train_rfe, y_train)
            score = model.score(X_test_rfe, y_test)
            score_list.append(score)
            if(score>high_score):
                high_score = score
                nof = nof_list[n]

        print("Optimum number of features: %d with score: %f" % (nof, high_score))

        cols = list(x.columns)
        model = LinearRegression()
        rfe = RFE(model, nof)
        X_rfe = rfe.fit_transform(x,y)
        model.fit(X_rfe,y)
        temp = pd.Series(rfe.support_,index = cols)
        selected_features_rfe = temp[temp==True].index
        pickle.dump(model, open(file_name, 'wb'))

        with open('parameters_selection.txt', 'w') as f:
            for item in selected_features_rfe:
                f.write("%s\n" % item)

        return selected_features_rfe

    def drop_col_feat_imp(model, X_train, y_train, random_state = 42):

        # clone the model to have the exact same specification as the one initially trained
        model_clone = clone(model)
        # set random_state for comparability
        model_clone.random_state = random_state
        # training and scoring the benchmark model
        model_clone.fit(X_train, y_train)
        benchmark_score = model_clone.score(X_train, y_train)
        # list for storing feature importances
        importances = []

        # iterating over all columns and storing feature importance (difference between benchmark and new model)
        for col in X_train.columns:
            model_clone = clone(model)
            model_clone.random_state = random_state
            model_clone.fit(X_train.drop(col, axis = 1), y_train)
            drop_col_score = model_clone.score(X_train.drop(col, axis = 1), y_train)
            importances.append(benchmark_score - drop_col_score)

        importances_df = pd.DataFrame(data=importances, index=X_train.columns.tolist(), columns=['importance'])
        importances_df.sort_values(by='importance', inplace=True, ascending=False)

        return importances_df


    def permutation_importance(model, X_train, y_train):
        def perm(X_train, y_train):
            perm = PermutationImportance(model, cv = None, refit = False, n_iter = 50).fit(X_train, y_train)
            return perm.feature_importances_

        imp_perm = perm(X_train, y_train)

        feature_df = pd.DataFrame(data=imp_perm, index=X_train.columns.tolist(), columns=['importance'])
        feature_df.sort_values(by='importance', inplace=True)

        return feature_df

    def Embedded_Method(self, x, y, plot_matr = 'yes'):
        reg = LassoCV()
        reg.fit(x, y)
        print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
        print("Best score using built-in LassoCV: %f" %reg.score(x,y))
        coef = pd.Series(reg.coef_, index = x.columns)
        print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
        imp_coef = coef.sort_values()
        if plot_matr == 'yes':
            matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
            imp_coef.plot(kind = "barh")
            plt.title("Feature importance using Lasso Model")
            plt.show()
        return imp_coef


        # ███    ███  ██████  ██████  ███████ ██      ███████
        # ████  ████ ██    ██ ██   ██ ██      ██      ██
        # ██ ████ ██ ██    ██ ██   ██ █████   ██      ███████
        # ██  ██  ██ ██    ██ ██   ██ ██      ██           ██
        # ██      ██  ██████  ██████  ███████ ███████ ███████


    def modelDTR(self, x_train, x_test, y_train, y_test):
        def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
            model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
            model.fit(train_X, train_y)
            preds_val = model.predict(val_X)
            mae = mean_absolute_error(val_y, preds_val)
            return mae, max_leaf_nodes
        leafs = [500, 1000, 2500, 5000, 10000, 25000, 50000]
        my_mae = [get_mae(leafs[i], x_train, x_test, y_train, y_test) for i in range(len(leafs))]
        return min(my_mae)

    def XGB_tuning(self, x_train, x_test, y_train, y_test, n_estimator = 500, learning_rate = 0.05, early_stopping_rounds = 5, n_cores = 2, file_name = 'model.sav'):
        model = XGBRegressor(n_estimator = n_estimator, learning_rate = learning_rate, n_jobs = n_cores)
        model.fit(x_train, y_train, early_stopping_rounds = early_stopping_rounds, eval_set = [(x_test, y_test)], verbose = False)
        pickle.dump(model, open(file_name, 'wb'))

    def casual_model(self, x_train, y_train, model, file_name = 'model.sav'):
        model.fit(x_train, y_train)
        pickle.dump(model, open(file_name, 'wb'))

    def cross_validation_method(self, model, x, y, fold = 5):
        score = cross_val_score(model, x, y, cv=fold)
        return score

        # ██████  ███████ ███████ ██    ██ ██   ████████ ███████
        # ██   ██ ██      ██      ██    ██ ██      ██    ██
        # ██████  █████   ███████ ██    ██ ██      ██    ███████
        # ██   ██ ██           ██ ██    ██ ██      ██         ██
        # ██   ██ ███████ ███████  ██████  ███████ ██    ███████

    def play_model(self, x_test, y_test, file_name = 'model.sav'):
        model = pickle.load(open(file_name, 'rb'))
        preds_val = model.predict(x_test)
        mae = mean_absolute_error(y_test, preds_val)
        score = model.score(x_test, y_test)# is comparing the predictions of the model against the real labels
        # score2 = accuracy_score(y_test, preds_val)
        return mae, score
