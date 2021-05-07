palette = ["#1F77B4","#FF7F0E","#2CA02C", "#00A3E0", '#4943cf', '#1eeca8', '#e52761', '#490b04', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b',
                   '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff',
                   '#a3c1ad', '#a0d6b4', '#5f9ea0', '#317873', '#49796b', '#ffb3ba', '#ffdfba', '#d0d04a', '#baffc9', '#bae1ff', '#a3c1ad', '#a0d6b4']
import sys
sys.path.insert(0, 'C:/Users/Max Power/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
from AJ_draw import disegna as ds

# train a generative adversarial network on a one-dimensional function
# from keras.models import Sequential
# from keras.layers import Dense, LeakyReLU, Reshape, Flatten, Conv1D, Dropout
# from keras.models import load_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LeakyReLU, Dropout, Reshape, Conv1DTranspose, Flatten

import numpy as np
import pandas as pd

class learning_gan:

    # ███    ███  ██████  ██████  ███████ ██      ███████
    # ████  ████ ██    ██ ██   ██ ██      ██      ██
    # ██ ████ ██ ██    ██ ██   ██ █████   ██      ███████
    # ██  ██  ██ ██    ██ ██   ██ ██      ██           ██
    # ██      ██  ██████  ██████  ███████ ███████ ███████

    # define the standalone discriminator model
    def define_discriminator(self, n_inputs=2):
        '''
        modello discriminatore, testa del modello, impara a distinguere tra il dato reale e quello generato.
        n_inputs: (int) dimensione del dato reale
        return model: il modello
        '''
        model = Sequential()
        model.add(Dense(int(n_inputs/2), activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
        model.add(Dropout(0.4))
        model.add(Dense(int(n_inputs/10), activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(0.4))
        model.add(Dense(int(n_inputs/100), activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # define the standalone generator model
    def define_generator(self, latent_dim, n_outputs=2):
        '''
        modello generatore, coda del modello, impara a creare un dato che possa essere confuso con un dato reale.
        latent_dim: (int) dimensione dello spazio latente, che sono numeri random
        n_outputs: (int) dimensione del dato reale, questo e' il dato generato che vuole imitare il dato reale
        return: model
        '''
        model = Sequential()
        model.add(Dense(int(n_outputs*1.5), activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
        model.add(Dropout(0.4))
        model.add(Dense(int(n_outputs/10), activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(0.4))
        model.add(Dense(int(n_outputs/100), activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(n_outputs, activation='linear'))
        return model

    # define the standalone generator model
    def define_generator_tf2(latent_dim, n_outputs=2):
        '''
        questo modello funziona solo con TF2
        modello generatore, coda del modello, impara a creare un dato che possa essere confuso con un dato reale.
        latent_dim: (int) dimensione dello spazio latente, che sono numeri random
        n_outputs: (int) dimensione del dato reale, questo e' il dato generato che vuole imitare il dato reale
        return: model
        '''

        model = Sequential()

        # foundation for 7x7 image
        n_nodes = 128 * int(n_outputs/4)

        model.add(Dense(n_nodes, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((int(n_outputs/4), 128)))

        # upsample to 14x14
        model.add(Conv1DTranspose(128, 4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        # upsample to 28x28
        model.add(Conv1DTranspose(128, 4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv1D(1, int(n_outputs/4), activation='sigmoid', padding='same'))
        model.add(Flatten())
        return model

    # define the combined generator and discriminator model, for updating the generator
    def define_gan(self, generator, discriminator):
        '''
        modello completo di GAN, generato concatenando il generatore e il discriminatore.
        Il disciminatore ha i pesi fissato, cosi che durante il training solo il generatore viene aggiornato
        generator: modello del generatore
        discriminator: modello del discriminatore
        return: modello
        '''
        # make weights in the discriminator not trainable
        discriminator.trainable = False
        # connect them
        model = Sequential()
        # add generator
        model.add(generator)
        # add the discriminator
        model.add(discriminator)
        # compile model
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model


    # ██████   █████  ████████  █████       ██████  ███████ ███    ██ ███████ ██████   █████  ████████  ██████  ██████
    # ██   ██ ██   ██    ██    ██   ██     ██       ██      ████   ██ ██      ██   ██ ██   ██    ██    ██    ██ ██   ██
    # ██   ██ ███████    ██    ███████     ██   ███ █████   ██ ██  ██ █████   ██████  ███████    ██    ██    ██ ██████
    # ██   ██ ██   ██    ██    ██   ██     ██    ██ ██      ██  ██ ██ ██      ██   ██ ██   ██    ██    ██    ██ ██   ██
    # ██████  ██   ██    ██    ██   ██      ██████  ███████ ██   ████ ███████ ██   ██ ██   ██    ██     ██████  ██   ██


    # generate n real samples with class labels
    def generate_real_samples(self, data, n):
        '''
        selezione un sotto campione random dei dati.
        data: dati del campione, in formato Pandas
        n: (int) numero di campioni da selezionare
        return: data_selection, pandas dataframe con n campioni dentro, y, numpy array con tutti 1 e n campioni dentro
        (y rappresenta la classe dei dati, 1 indica che sono quelli reali)
        '''
        data_selection = data.sample(n).reset_index(drop=True)
        y = np.ones((n, 1))
        return data_selection, y

    # generate points in latent space as input for the generator
    def generate_latent_points(self, latent_dim, n):
        '''
        genera punti dallo spazio latente. Altro non sono che valori con distribuzione gaussiana
        latent_dim: (int) dimensione del vettore da dare in pasto al generatore
        n: (int) numero di campioni da generare
        return: x_input, numpy array con n campioni di latent_dim valori random
        '''
        # generate points in the latent space
        x_input = np.random.randn(latent_dim * n)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n, latent_dim)
        return x_input

    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, generator, latent_dim, n):
        '''
        generatore di campioni falsi, usa il modello generatore per predirre un nuovo set di dati che quindi sono inventati
        generator: modello del generatore
        latent_dim: (int) dimensione del vettore dello spazio latente, deve essere lungo quanto il primo layer del modello
        n: (int) numero di campioni da generare
        return: x, numpy array con il dato inventato, y numpy array con tutti 0 dentro e lungo n, rappresenta il label dei dati finti
        '''
        # generate points in latent space
        x_input = self.generate_latent_points(latent_dim, n)
        # predict outputs
        X = generator.predict(x_input)
        # create class labels
        y = np.zeros((n, 1))
        return X, y


    # ████████ ██████   █████  ██ ███    ██ ██ ███    ██  ██████
    #    ██    ██   ██ ██   ██ ██ ████   ██ ██ ████   ██ ██
    #    ██    ██████  ███████ ██ ██ ██  ██ ██ ██ ██  ██ ██   ███
    #    ██    ██   ██ ██   ██ ██ ██  ██ ██ ██ ██  ██ ██ ██    ██
    #    ██    ██   ██ ██   ██ ██ ██   ████ ██ ██   ████  ██████


    # evaluate the discriminator and plot real and fake points
    def summarize_performance(self, data, epoch, generator, discriminator, latent_dim, verbose=False):
        '''
        funzione per fare lo scoring del training
        data: training set in formato pandas
        epoch: (int) epoca del training
        generator: modello del generatore
        discriminator: modello del discriminatore
        latent_dim: (int) dimensione del vettore dello spazio latente, deve essere lungo quanto il primo layer del modello
        verbose: boolean, if True print accuracy and plot
        return: epoch, acc_real (accuratezza nel identificare i dati reali), acc_fake (accuratezza nel identificare i dati fake)
        '''
        # prepare real samples
        x_real, y_real = self.generate_real_samples(data, 100)
        # evaluate discriminator on real examples
        _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
        # prepare fake examples
        x_fake, y_fake = self.generate_fake_samples(generator, latent_dim, 100)
        # evaluate discriminator on fake examples
        _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
        # summarize discriminator performance

        x_real, y_real = self.generate_real_samples(data, 1)
        x_fake, y_fake = self.generate_fake_samples(generator, latent_dim, 1)

        if verbose:
            print('epoch: ',epoch)
            print('how good the discriminator is to evaluate the real example',acc_real)
            print('how good the discriminator is to evaluate the fake example',acc_fake)
            # scatter plot real and fake data points
            ds().nuova_fig(1)
            X1 = np.arange(len(x_real.T[0]))
            ds().dati(x = X1, y = x_real.T[0], colore = palette[0], descrizione="real")
            ds().dati(x = X1, y = x_fake.T, colore = palette[1], descrizione="fake")
            ds().legenda()
            ds().porta_a_finestra()
        return epoch, acc_real, acc_fake

    # train the generator and discriminator
    def train(self, data, latent_dim, n_epochs=10000, n_batch=128, n_eval=2000, verbose = False, save = False):
        '''
        training function.
        data: training set in formato pandas
        latent_dim: (int) dimensione del vettore dello spazio latente, deve essere lungo quanto il primo layer del modello
        n_epochs: (int) numero totale di epoche per il training
        n_batch: (int) numero di campioni da usare per il training, se usato un numero maggiore alla dimensione di data, il valore viene convertito alla lunghezza di data
        n_eval: (int) ogni quante epoche viene salvato il modello e valutata l'accuratezza
        verbose: (boolean) if True viene printata l'accuratezza ogni n_eval e generato il plot
        save: (boolean) if True viene salvato il modello ogni n_eval
        '''
        size_single_sample = data.shape[1]
        # # create the discriminator
        discriminator = self.define_discriminator(n_inputs = size_single_sample)
        # # create the generator
        generator = self.define_generator(latent_dim, n_outputs = size_single_sample)
        # # create the gan
        gan_model = self.define_gan(generator, discriminator)

        # determine half the size of one batch, for updating the discriminator
        if n_batch>data.shape[1]:
            n_batch = data.shape[1]
        half_batch = int(n_batch / 2)
        sum_epoch, sum_acc_real, sum_acc_fake = [], [], []
        sum_d_loss_real, sum_d_loss_fake, sum_g_loss = [], [], []
        # manually enumerate epochs
        for i in range(n_epochs):
            # prepare real samples
            x_real, y_real = self.generate_real_samples(data, half_batch)
            # prepare fake examples
            x_fake, y_fake = self.generate_fake_samples(generator, latent_dim, half_batch)
            # update discriminator
            d_loss_real = discriminator.train_on_batch(x_real, y_real)
            d_loss_fake = discriminator.train_on_batch(x_fake, y_fake)
            # prepare points in latent space as input for the generator
            x_gan = self.generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator✬s error
            g_loss = gan_model.train_on_batch(x_gan, y_gan)
            # evaluate the model every n_eval epochs
            if (i+1) % n_eval == 0:
                temp_epoch, temp_real, temp_fake = self.summarize_performance(data, i, generator, discriminator, latent_dim, verbose)
                sum_epoch.append(temp_epoch)
                sum_acc_real.append(temp_real)
                sum_acc_fake.append(temp_fake)
                sum_d_loss_real.append(d_loss_real[0])
                sum_d_loss_fake.append(d_loss_fake[0])
                sum_g_loss.append(g_loss)
                if save:
                    self.save_model(generator, filename = 'generator_model_%03d.h5' % (i + 1))
        ds().nuova_fig(4)
        ds().dati(x = sum_epoch, y = sum_acc_real, colore=palette[0], descrizione='acc real')
        ds().dati(x = sum_epoch, y = sum_acc_fake, colore=palette[1], descrizione='acc fake')
        ds().legenda()
        ds().porta_a_finestra()

        ds().nuova_fig(5)
        ds().dati(x = sum_epoch, y = sum_d_loss_real, colore=palette[0], descrizione='loss dis real')
        ds().dati(x = sum_epoch, y = sum_d_loss_fake, colore=palette[1], descrizione='loss dis fake')
        ds().legenda()
        ds().porta_a_finestra()

        ds().nuova_fig(5)
        ds().dati(x = sum_epoch, y = sum_g_loss, colore=palette[2], descrizione='loss gan')
        ds().legenda()
        ds().porta_a_finestra()


        # ███████  █████  ██    ██ ███████
        # ██      ██   ██ ██    ██ ██
        # ███████ ███████ ██    ██ █████
        #      ██ ██   ██  ██  ██  ██
        # ███████ ██   ██   ████   ███████


    def save_model(self, model, filename = 'model'):
        '''
        salva il modello in un formato h5
        model: modello
        filename: nome del file
        '''
        model.save(filename)

    def load_model(self, filename):
        '''
        load il modello
        filename: nome del file con estensione
        '''
        return load_model(filename)
