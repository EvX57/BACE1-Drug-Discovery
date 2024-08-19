# Implements QSAR-NN models

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError, MeanAbsoluteError
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import statistics

from Vocabulary import Vocabulary
from AE import Autoencoder

class Predictor():
    def __init__(self, path, property, load, split, vocab, autoencoder, df, suffix='', hyperparams=[1,256,0.0001], input_length=256):
        self.path = path
        self.property = property
        self.dropout = 0.3
        self.n_layers = hyperparams[0]
        self.n_units = hyperparams[1]
        self.learning_rate = hyperparams[2]
        self.n_epochs = 1000
        self.batch_size = 32
        self.validation_split = 0.1
        self.load = load
        self.split = split
        self.vocab = vocab
        self.auto = autoencoder
        self.data = df
        self.input_length = input_length

        if not load:
            self.get_latent_representations()
            self.train_test_split()
        self.build_model()

        if self.load:
            self.load_model(self.property, suffix)
    
    # Convert SELFIES to latent vector
    def selfies_to_latentvector(self, selfies):
        return self.auto.selfies_to_latentvector(self.vocab, selfies)
    
    # Add column for latent representations into self.data dataframe
    def get_latent_representations(self):
        selfies = list(self.data['selfies'])
        lat_vecs = self.selfies_to_latentvector(selfies).tolist()
        self.data['LV'] = lat_vecs

    # Create train and test data
    # 0.8-0.1-0.1 train-validation-test split
    def train_test_split(self):
        # Shuffle dataframe
        self.data = self.data.sample(frac=1, ignore_index=True)

        # Create X and Y train
        lat_vecs = list(self.data['LV'])
        property = list(self.data[self.property])

        self.range = max(property) - min(property)

        train_length = int(len(lat_vecs) * self.split)
        self.X_train = np.array(lat_vecs[:train_length])
        self.Y_train = np.array(property[:train_length])
        self.X_test = np.array(lat_vecs[train_length:])
        self.Y_test = np.array(property[train_length:])

        # Get input length from latent vector
        self.input_length = len(self.X_train[0])

    # Create model
    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.input_length)))
        for _ in range(self.n_layers):
            model.add(Dense(self.n_units, activation='relu'))
            model.add(Dropout(rate=self.dropout))
        model.add(Dense(1, activation='linear'))

        self.model = model
        opt = Adam(learning_rate=self.learning_rate)
        self.model.compile(loss=MSE, optimizer=opt, metrics=[RootMeanSquaredError(), MeanAbsolutePercentageError(), MeanAbsoluteError()])

    # Compile and train model          
    def train_model(self):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20, restore_best_weights=True)
        mc = ModelCheckpoint(self.path + 'best_model_' + self.property + '.h5', monitor='val_loss', mode='min', verbose=0, save_best_only=True)
        
        result = self.model.fit(self.X_train, self.Y_train, epochs=self.n_epochs, batch_size=self.batch_size, validation_split=self.validation_split, callbacks = [es, mc], verbose=0)
        
        # Training curve for MSE
        plt.plot(result.history['loss'], label='Train')
        plt.plot(result.history['val_loss'], label='Validation')
        plt.title('Training Loss')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig(self.path + 'training_loss_' + self.property + '.png')
        plt.close()

        # Training curve for RMSE
        plt.plot(result.history['root_mean_squared_error'], label='Train')
        plt.plot(result.history['val_root_mean_squared_error'], label='Validation')
        plt.title('Training RMSE')
        plt.ylabel('RMSE')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig(self.path + 'training_rmse_' + self.property + '.png')
        plt.close()

        # Save model
        self.model.save(self.path + 'model_' + self.property + '.h5')

        print('DONE')

    # Load pre-trained predictor model
    def load_model(self, model_name, suffix=''):
        self.model.load_weights(self.path + "model_" + model_name + suffix + ".h5")
    
    # Evaluate model performance
    # Input
    # X_test: SELFIES
    # Y_test: molecular property value
    def evaluate(self):
        performance = self.model.evaluate(self.X_test, self.Y_test)
        mse = performance[0]
        rmse = performance[1]
        mape = performance[2]
        mae = performance[3]

        # Determine correlation
        y_pred = np.array(self.predict(self.X_test, string=False))
        reg = LinearRegression().fit(self.Y_test.reshape((-1,1)), y_pred.reshape((-1,1)))
        r_sq = reg.score(self.Y_test.reshape((-1,1)), y_pred.reshape(-1,1))
        x_vals = np.arange(min(self.Y_test), max(self.Y_test), (max(self.Y_test)-min(self.Y_test))/100).reshape((-1,1))
        y_vals = reg.predict(x_vals)
        x_vals = x_vals.reshape(len(x_vals))
        y_vals = y_vals.reshape(len(y_vals))

        # Write results to file
        results = open(self.path + 'evaluation_' + self.property + '.txt', 'w')
        results.write('MSE: ' + str(round(mse, 4)))
        results.write('\nRMSE: ' + str(round(rmse, 4)))
        results.write('\nMAPE: ' + str(round(mape, 4)))
        results.write('\nMAE: ' + str(round(mae, 4)))
        results.write('\nRange: ' + str(round(self.range, 4)))
        results.write('\nPercent Error: ' + str(round(100*mae/self.range, 4)))
        results.write('\nR-Squared: ' + str(round(r_sq, 4)))
        results.close()

        # Plot correlation graph        
        # Scatter plot
        plt.scatter(self.Y_test, y_pred, s=7.5)  # 2.5
        plt.plot(x_vals, y_vals, color='black', linestyle='dashed', linewidth=2)
        plt.xlabel('Expected', fontsize=11)
        plt.ylabel('Predicted', fontsize=11)
        
        if self.property == 'logCmax':
            plt.title('log(' + r'$C_{max}$' + ') QSAR-NN', fontsize=15)
        elif self.property == 'logThalf':
            plt.title('log(' + r'$t_{1/2}$' + ') QSAR-NN', fontsize=15)
        else:
            plt.title(self.property + ' QSAR-NN', fontsize=15)
        plt.tight_layout()
        plt.savefig(self.path + 'correlation_scatter_' + self.property + '.png')
        plt.close()

        return mse, rmse

    # Make predictions for molecular property
    # Input: SELFIES or latent vectors (depending on string boolean)
    # Output: molecular property
    def predict(self, selfies, string=True):
        if string:
            lat_vecs = self.selfies_to_latentvector(selfies)
            predictions = self.model.predict(lat_vecs)
        else:
            lat_vecs = selfies
            predictions = self.model(lat_vecs)
        
        predictions = [p[0] for p in predictions]
        
        if tf.is_tensor(predictions[0]):
            predictions = [p.numpy() for p in predictions]
        
        return predictions

# Optimize num layers, num hidden units per layer, learning rate, batch size
def optimize_hyperparameters(path, property, df, vocab, auto):
    n_layers = [1, 2, 3]
    n_hidden_units = [128, 256, 512, 1024]
    learning_rate = [0.01, 0.001, 0.0001, 0.00001]

    # Num trials for each set of hyperparameters
    # Trials are combined for average and stdev to compare between hyperparameters
    n_trials = 5

    # Run all trials
    counter = 0
    hyperparam_sets = []
    for nl in n_layers:
        for nhu in n_hidden_units:
            for lr in learning_rate:
                predictor = Predictor(path, property, False, 0.8, vocab, auto, df, hyperparams=[nl, nhu, lr])
                trials = []
                for t in range(n_trials):
                    predictor.train_test_split()
                    predictor.build_model()
                    predictor.train_model()
                    mse, _ = predictor.evaluate()
                    trials.append(mse)
                    counter += 1
                    print(str(counter) + ' Done')
                hyperparam_sets.append([nl, nhu, lr, statistics.mean(trials), statistics.stdev(trials)])
    
    # Save results
    new_df = pd.DataFrame(data=hyperparam_sets, columns=['Layers', 'Hidden Units', 'Learning Rate', 'Mean', 'Stdev'])
    new_df.to_csv(path + 'hyperparam_opt.csv', index=False)

# Train the predictor on a specific molecular property / metric
def run_target(path, df_path, property):
    vocab_df = pd.read_csv('datasets/500k_small_molecule.csv')
    ae_path = 'models/AE_model.h5'
    df = pd.read_csv(df_path)

    vocab = Vocabulary(list(vocab_df['selfies']))

    latent_dim = 256
    embedding_dim = 256
    lstm_units = 512
    batch_size = 128
    batch_norm = True
    batch_norm_momentum = 0.9
    numb_dec_layer = 2
    noise_std = 0.1
    input_shape = (vocab.max_len, vocab.vocab_size)
    output_dim = vocab.vocab_size
    auto = Autoencoder(path, input_shape, latent_dim, lstm_units, output_dim, batch_norm, batch_norm_momentum, noise_std, numb_dec_layer, embedding_dim, vocab.vocab_size, vocab.max_len)
    auto.load_autoencoder_model(ae_path)

    predictor = Predictor(path, property, False, 0.8, vocab, auto, df, suffix='_500k', hyperparams=[1,256,0.0001]) # pIC50
    #predictor = Predictor(path, property, False, 0.8, vocab, auto, df, suffix='_500k', hyperparams=[1,256,0.001]) # logBB
    #predictor = Predictor(path, property, False, 0.8, vocab, auto, df, suffix='_500k', hyperparams=[1,1024,0.0001]) # logCmax
    #predictor = Predictor(path, property, False, 0.8, vocab, auto, df, suffix='_500k', hyperparams=[1,512,0.0001]) # logT1/2

    predictor.train_model()
    predictor.evaluate()
    
    #optimize_hyperparameters(path, property, df, vocab, auto)

# Predict values for an input metric
def predict_metric(path, property, vocab, auto, df_train, df_repurpose, hyperparams, save_path, str=True, lv=None):
    # Load predictor
    predictor = Predictor(path, property, True, 0.8, vocab, auto, df_train, hyperparams=hyperparams)

    # Make predictions
    print('Predictions starting...')
    if str:
        all_selfies = list(df_repurpose['selfies'])
        predictions = predictor.predict(all_selfies, string=True)
    else:
        predictions = predictor.predict(lv, string=False)
        
    print('Predictions complete!')
    df_repurpose[property] = predictions
    df_repurpose.to_csv(save_path, index=False)

if __name__ == "__main__":
    path = 'test/'
    df_path = 'datasets/BACE1.csv'
    property = 'pIC50'
    run_target(path, df_path, property)