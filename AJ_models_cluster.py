from AJ_draw import disegna as ds
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture

# link with explanation
# https://scikit-learn.org/stable/modules/clustering.html

class learning_cluster:
    def __init__ (self, SEED = 222):
        self.SEED = SEED
        self.n_clusters = 2


        # ███    ███  ██████  ██████  ███████ ██      ███████
        # ████  ████ ██    ██ ██   ██ ██      ██      ██
        # ██ ████ ██ ██    ██ ██   ██ █████   ██      ███████
        # ██  ██  ██ ██    ██ ██   ██ ██      ██           ██
        # ██      ██  ██████  ██████  ███████ ███████ ███████

    def get_models(self, list_chosen):
        """Generate a library of base learners.
        :param list_chosen: list with the names of the models to load
        :return: models, a dictionary with as index the name of the models, as elements the models"""


        ap = AffinityPropagation(damping = 0.9)
        ac = AgglomerativeClustering(n_clusters=self.n_clusters)
        bi = Birch(threshold=0.01, n_clusters=self.n_clusters)
        db = DBSCAN(eps=0.30, min_samples=9)
        km = KMeans(n_clusters=self.n_clusters)
        mkm = MiniBatchKMeans(n_clusters=self.n_clusters)
        ms = MeanShift()
        op = OPTICS(eps=0.8, min_samples=10)
        sc = SpectralClustering(n_clusters=self.n_clusters)
        gm = GaussianMixture(n_components=self.n_clusters)

        models_temp = {
                  'AffinityPropagation': ap,
                  'AgglomerativeClustering': ac,
                  'Birch': bi,
                  'DBSCAN': db,
                  'kMeans': km,
                  'miniKMeans': mkm,
                  'MeanShift': ms,
                  'OPTICS': op,
                  'SpectralClustering': sc,
                  'GaussianMixture': gm,
                  }

        models = dict()
        for model in list_chosen:
            if model in models_temp:
                models[model] = models_temp[model]
        return models


        # ████████ ██████   █████  ██ ███    ██ ██ ███    ██  ██████
        #    ██    ██   ██ ██   ██ ██ ████   ██ ██ ████   ██ ██
        #    ██    ██████  ███████ ██ ██ ██  ██ ██ ██ ██  ██ ██   ███
        #    ██    ██   ██ ██   ██ ██ ██  ██ ██ ██ ██  ██ ██ ██    ██
        #    ██    ██   ██ ██   ██ ██ ██   ████ ██ ██   ████  ██████

    def train_models(self, models, xtrain):
        '''training function
        :param models: dictionary with as index the name of the models, as elements the models
        :param xtrain: matrix with the features, pandas
        :return: models, a dictionary with as index the name of the models, as elements the models after the training
        '''

        for i, (name_model, model) in enumerate(models.items()):
            if name_model != 'AgglomerativeClustering' or name_model != 'DBSCAN' or name_model != 'MeanShift' or name_model != 'OPTICS' or name_model != 'SpectralClustering':
                model.fit(xtrain)
        return models



        # ███████  ██████  ██████  ███████  ██████  █████  ███████ ████████
        # ██      ██    ██ ██   ██ ██      ██      ██   ██ ██         ██
        # █████   ██    ██ ██████  █████   ██      ███████ ███████    ██
        # ██      ██    ██ ██   ██ ██      ██      ██   ██      ██    ██
        # ██       ██████  ██   ██ ███████  ██████ ██   ██ ███████    ██


    def cluster_matrix_generator(self, models, xtest):
        '''generate the prediction and the probabilities with all the models in the list and add all of them to the same matrix, adding the average prediciton (ensamble)
        :param models: dictionary with as index the name of the models, as elements the models
        :param xtest: matrix with the features, pandas
        :return: y_pred_matrix.shape[0] = xtest.shape[0], y_pred_matrix.shape[1] = len(models), Pandas matrix with the predicion for all the models
        '''

        y_pred_matrix = pd.DataFrame()
        for i, (name_model, model) in enumerate(models.items()):
            if name_model == 'AgglomerativeClustering' or name_model == 'DBSCAN' or name_model == 'MeanShift' or name_model == 'OPTICS' or name_model == 'SpectralClustering':
                y_pred_matrix[name_model] = model.fit_predict(xtest)
            else:
                y_pred_matrix[name_model] = model.predict(xtest)
            # y_pred_matrix['Ensamble'] = round(y_pred_matrix.mean(axis=1)).astype(int)
        return y_pred_matrix



        # ████████ ██    ██ ███    ██ ██ ███    ██  ██████
        #    ██    ██    ██ ████   ██ ██ ████   ██ ██
        #    ██    ██    ██ ██ ██  ██ ██ ██ ██  ██ ██   ███
        #    ██    ██    ██ ██  ██ ██ ██ ██  ██ ██ ██    ██
        #    ██     ██████  ██   ████ ██ ██   ████  ██████

    def elbow_method(self, df):
        sse={}
        for k in range(1, 10):
            kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df)
            sse[k] = kmeans.inertia_
        plt.figure()
        plt.plot(list(sse.keys()), list(sse.values()))
        plt.xlabel("Number of cluster")
        plt.show()

    def order_cluster(self, cluster_field_name, target_field_name, df, ascending):
        '''
        function for ordering cluster numbers
        visto che kmean quando crea le classificazioni non e' ordinato secondo nessun criterio particolare
        con questa funzione il database rinomina le classi in modo che siano ordinate secondo il target
        '''

        df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
        df_new = df_new.sort_values(by=target_field_name, ascending=ascending).reset_index(drop=True)
        df_new['index'] = df_new.index
        df_final = pd.merge(df, df_new[[cluster_field_name,'index']], on=cluster_field_name)
        df_final = df_final.drop([cluster_field_name],axis=1)
        df_final = df_final.rename(columns={"index":cluster_field_name})
        return df_final
