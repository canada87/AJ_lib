palette = ["#1F77B4","#FF7F0E","#2CA02C", "#00A3E0", '#4943cf', '#1eeca8', '#e52761', '#490b04', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b',
                   '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff',
                   '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4']
import sys
sys.path.insert(0, 'C:/Users/Max Power/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
sys.path.insert(0, 'C:/Users/ajacassi/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
from AJ_draw import disegna as ds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error
import numpy as np
import random
from tqdm import tqdm
from math import log

class Genetic_Algorithm:
    def __init__(self, total_models, epsilon, nodes, regression = True):
        '''
        total_models: popolazione di modelli usati per competere geneticamente (devono essere un numero pari)
        epsilon: con che intensita viene applicata una mutazione, piu il valore e' grande piu la rete cambia durante la mutazione ma e' anche piu instabile. Piu e' piccolo piu e' stabile ma richiede piu tempo per convergere.
        nodes: vettore con il numero di nodi della rete neurale, il primo e' il numero di input e l'ultimo e' il numero di output
        regression: boolean, se true usa MAE come loss function se False usa CE come loss function
        '''
        self.total_models = total_models
        self.variation = epsilon
        self.nodes = nodes
        self.regression = regression

        self.highest_fitness = -1
        self.current_pool = []
        self.fitness = []
        self.predictions = []
        self.best_weights = []

        # Initialize all models
        for i in range(self.total_models):
            model = self.create_model(self.nodes)
            self.current_pool.append(model)
            self.fitness.append(-100)
            self.predictions.append(0)
        self.best_weights = model.get_weights()
        self.highest_fitness = self.fitness[0]

    def create_model(self, nodes):
        '''
        crea il modello di rete neurale
        nodes: vettore con il numero di nodi della rete neurale, il primo e' il numero di input e l'ultimo e' il numero di output
        return: il modello
        '''
        model = Sequential()
        model.add(Dense(nodes[1], input_dim=nodes[0], activation='relu'))
        for i in range(2, len(nodes)-1):
            model.add(Dense(nodes[i], activation='relu'))
        if self.regression:
            model.add(Dense(nodes[-1], activation='linear'))
        else:
            model.add(Dense(nodes[-1], activation='softmax'))
        return model

    def predict_action(self, x, model_num):
        '''
        funzione per fare la predizione sul modello scelto
        x: features di treaning
        model_num: indice del modello che si vuole usare per la predizione dall'insieme dei modelli
        return: predizione
        '''
        output_prob = self.current_pool[model_num].predict(x)
        return output_prob

    def fitness_func(self, y_true, y_pred):
        '''
        funzione che calcola la bunta della soluzione
        y_true: target reale
        y_pred: target predetto
        return: valore di fitness, piu e' grande piu il modello e' buono
        '''
        # calculate cross entropy
        def cross_entropy(p, q):
        	return -sum([p[i]*log(q[i]) for i in range(len(p))])

        if self.regression:
            mae = mean_absolute_error(y_true, y_pred)
        else:
            ce = map(cross_entropy, y_true, y_pred)
            mae = np.array(list(ce)).mean()
        abs_error = mae + 0.00000001
        solution_fitness = 1.0 / abs_error
        return solution_fitness

    def model_breeding(self, parent1, parent2):
        '''
        funzione che genera i modelli figli dai 2 genitori. Presi 2 genitori, viene scelto un valore a caso tra i pesi delle 2 reti e vengono scambiati. Ne risutano 2 nuove reti uguali alle precedenti tranne per il gene che e' stato scambiato con l'altro genitore
        parent1: indice del modello 1
        parent2: indice del modello 2
        return: 2 modelli con un gene modificato rispetto alla partenza
        '''
        weight1 = self.current_pool[parent1].get_weights()
        weight2 = self.current_pool[parent2].get_weights()
        new_weight1 = weight1.copy()
        new_weight2 = weight2.copy()
        gene = random.randint(0,len(new_weight1)-1)
        new_weight1[gene] = weight2[gene]
        new_weight2[gene] = weight1[gene]
        return new_weight1, new_weight2

    def model_mutate(self, weights):
        '''
        funzione che applica una mutazione a tutti i pesi della rete
        weight: sono i pesi della rete su cui si vuole applicare la mutazione
        return: pesi della rete dopo la mutazione
        '''
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                if(random.uniform(0,1) > .85):
                    change = random.uniform(-self.variation, self.variation)
                    weights[i][j] += change
        return weights

    def generational_ark(self, x, y):
        '''
        funzione che genera le predizione con i modelli scelti, calcola la bonta dei modelli, seleziona i 2 piu performanti, genera tanti figli quanti il numero di modelli originali,
        applica mutazioni su tutti i figli. Nessuna delle reti ha performato meglio della migliore in assoluto, viene usata la migliore come genitore. Se invece performano meglio della migliore,
        la migliore viene aggiornata con i modelli recenti.
        x: features
        y: targets
        return: fit min, fit max, fit medio, ultimo moglior fit, pesi della miglir rete
        '''
        for i in range(self.total_models):
            self.predictions[i] = self.predict_action(x, i)
            self.fitness[i] = self.fitness_func(y, self.predictions[i])

        fitness_temp = self.fitness.copy()
        i_parent1 = fitness_temp.index(max(fitness_temp))
        fit_par1 = max(fitness_temp)

        fitness_temp.pop(i_parent1)
        i_parent2 = fitness_temp.index(max(fitness_temp))
        fit_par2 = max(fitness_temp)

        compete = [fit_par1, fit_par2, self.highest_fitness]
        if compete.index(max(compete)) == 0:
            self.best_weights = self.current_pool[i_parent1].get_weights()
        elif compete.index(max(compete)) == 1:
            self.best_weights = self.current_pool[i_parent2].get_weights()

        new_weights = []

        for couples in range(self.total_models//2):
            if fit_par1 < self.highest_fitness:
                self.current_pool[i_parent1].set_weights(self.best_weights)
            if fit_par2 < self.highest_fitness:
                self.current_pool[i_parent2].set_weights(self.best_weights)

            offspring = self.model_breeding(i_parent1, i_parent2)
            mutated1 = self.model_mutate(offspring[0])
            mutated2 = self.model_mutate(offspring[1])
            new_weights.append(mutated1)
            new_weights.append(mutated2)

        for select in range(len(new_weights)):
            self.current_pool[select].set_weights(new_weights[select])

        self.highest_fitness = max(compete)
        return min(self.fitness), max(self.fitness), np.array(self.fitness).mean(), self.highest_fitness, self.best_weights

    def genetic_fit(self, epochs, x, y):
        '''
        funzione di fit
        epochs: numero di epoche/generazioni
        x: fetures
        y: targets
        return: storico della fit_function nelle epoche, miglio modello di sempre
        '''
        self.fit_min, self.fit_max, self.fit_mean, self.fit_top = [], [], [], []

        pbar = tqdm(range(epochs), position = 0, leave = True, ascii = True, unit='epochs')
        for i in pbar:
            fit1, fit2, fit3, fit4, best_weights = self.generational_ark(x, y)
            self.fit_min.append(fit1)
            self.fit_max.append(fit2)
            self.fit_mean.append(fit3)
            self.fit_top.append(fit4)

        best_model = self.create_model(self.nodes)
        best_model.set_weights(best_weights)
        self.best_model = best_model
        return self.fit_min, self.fit_max, self.fit_mean, self.fit_top, best_model

    def plot_fitness(self):
        ds().nuova_fig(1)
        ds().dati(x = np.arange(len(self.fit_min)), y = self.fit_min, colore=palette[0], descrizione='min')
        ds().dati(x = np.arange(len(self.fit_max)), y = self.fit_max, colore=palette[1], descrizione = 'max')
        ds().dati(x = np.arange(len(self.fit_mean)), y = self.fit_mean, colore=palette[2], descrizione='mean')
        ds().dati(x = np.arange(len(self.fit_top)), y = self.fit_top, colore=palette[3], descrizione='best_ever')

        ds().legenda()
        ds().porta_a_finestra()

    def predict(self, x):
        '''
        funzione per predirre il miglio modello trainato
        x: features
        return: predictions
        '''
        pred = self.best_model.predict(x)
        return pred
