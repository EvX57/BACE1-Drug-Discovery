from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate,LSTM, Bidirectional, Dense, Input, GaussianNoise, BatchNormalization, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model
import numpy as np
import tensorflow as tf
import selfies as sf

from Vocabulary import Vocabulary
from rdkit.Chem import MolFromSmiles
import pandas as pd
import random
import os

class Autoencoder:
    def __init__(self, model_path, input_shape, latent_dim, lstm_units, output_dim, batch_norm, batch_norm_momentum, noise_std, numb_dec_layer, emb_dim, vocab_size, max_len, write_model_arch=False):
        self.path = model_path 
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.output_dim = output_dim
        self.batch_norm = batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.noise_std = noise_std

        self.numb_dec_layer = numb_dec_layer

        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.write = write_model_arch

        self.build_selfies_to_latent_model()
        self.build_latent_to_states_model()
        self.build_states_to_selfies_model()
        self.build_model()
    
    # SELFIES --> latent representation
    def build_selfies_to_latent_model(self):
        # INPUT: embedded encoding (SHAPE)
        # OUTPUT: latent representation (SHAPE)
        encoder_inputs = Input(shape = (None,), name = 'encoder_inputs')
        x = Embedding(self.vocab_size, self.lstm_units//2)(encoder_inputs)  
        
        states_list = [] 
        states_reversed_list = []
        for i in range(self.numb_dec_layer):
            # Only one layer
            if self.numb_dec_layer == 1:
                encoder = Bidirectional(LSTM(self.lstm_units // 2, return_state = True, name = 'encoder'+str(i)+'_LSTM'))

                # Outputs (from both directions), hidden state, cell state, hidden state reversed, cell state reversed
                x, state_h, state_c, state_h_reverse, state_c_reverse = encoder(x)
                
                states_list.append(state_h)
                states_list.append(state_c)
                states_reversed_list.append(state_h_reverse)
                states_reversed_list.append(state_c_reverse)
            # More than one layer & not last layer
            elif i != self.numb_dec_layer-1:
                encoder = Bidirectional(LSTM(self.lstm_units // 2, return_sequences = True, return_state = True, name = 'encoder'+str(i)+'_LSTM'))

                # Outputs (from both directions), hidden state, cell state, hidden state reversed, cell state reversed
                x, state_h, state_c, state_h_reverse, state_c_reverse = encoder(x)
                
                states_list.append(state_h)
                states_list.append(state_c)
                states_reversed_list.append(state_h_reverse)
                states_reversed_list.append(state_c_reverse)
                
                if self.batch_norm:
                    x  = BatchNormalization(momentum = self.batch_norm_momentum, name = 'BN_'+str(i))(x)
            # More than one  layer & last layer
            else:
                encoder2 = Bidirectional(LSTM(self.lstm_units//2, return_state = True, name = 'encoder'+str(i)+'_LSTM'))

                # Don't need actual output because it is already captured in hidden state output
                _, state_h2, state_c2, state_h2_reverse, state_c2_reverse = encoder2(x)
                
                states_list.append(state_h2)
                states_list.append(state_c2)
                states_reversed_list.append(state_h2_reverse)
                states_reversed_list.append(state_c2_reverse)
        
        # All hidden and cell states from forward and backward directions from all layers
        complete_states_list = states_list + states_reversed_list
        states = Concatenate(axis = -1, name = 'concatenate')(complete_states_list)

        if self.batch_norm:
            states = BatchNormalization(momentum = self.batch_norm_momentum, name = 'BN_'+str(i+1))(states)

        latent_representation = Dense(self.latent_dim, activation = "relu", name = "Dense_relu_latent_rep")(states)

        if self.batch_norm:
            latent_representation = BatchNormalization(momentum = self.batch_norm_momentum, name = 'BN_latent_rep')(latent_representation)

        #Adding Gaussian Noise as a regularizing step during training
        latent_representation = GaussianNoise(self.noise_std, name = 'Gaussian_Noise')(latent_representation)

        self.selfies_to_latent_model = Model(encoder_inputs, latent_representation, name = 'selfies_to_latent_model')

        if self.write:
            with open(self.path + 'selfies_to_latent.txt', 'w') as f:
                self.selfies_to_latent_model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Latent --> initial decoder hidden and cell states for LSTM models
    def build_latent_to_states_model(self):
        # INPUT: latent representation
        # OUTPUT: list w/ 2 elements being 1) hidden states and 2) cell states
        latent_input = Input(shape =(self.latent_dim,), name = 'latent_input')

        #List that will contain the reconstructed states
        decoded_states = []
        for dec_layer in range(self.numb_dec_layer):
            # Hidden and cell states each have a dense layer for reconstruction
            name = "Dense_h_" + str(dec_layer)
            h_decoder = Dense(self.lstm_units, activation = "relu", name = name)(latent_input)

            name = "Dense_c_" + str(dec_layer)
            c_decoder = Dense(self.lstm_units, activation ="relu", name = name)(latent_input)

            if self.batch_norm:
                name = "BN_h_" + str(dec_layer)
                h_decoder = BatchNormalization(momentum = self.batch_norm_momentum, name = name)(h_decoder)

                name = "BN_c_" + str(dec_layer)
                c_decoder = BatchNormalization(momentum = self.batch_norm_momentum, name = name)(c_decoder)

            decoded_states.append(h_decoder)
            decoded_states.append(c_decoder)

        self.latent_to_states_model = Model(latent_input, decoded_states, name = 'latent_to_states_model')
        if self.write:
            with open(self.path + 'latent_to_states.txt', 'w') as f:
                self.latent_to_states_model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Hidden and cell states --> SELFIES
    def build_states_to_selfies_model(self):
        # INPUT: hidden and cell states & one hot encoding (teacher forcing)
        # OUTPUT: one hot encoding of predictions (next timestep based on input timesteps)

        #decoder inputs needed for teacher's forcing
        decoder_inputs = Input(shape = self.input_shape, name = "decoder_inputs")

        # One hot + states
        inputs = []
        inputs.append(decoder_inputs)
        x = decoder_inputs

        # Use respective hidden and cell state outputs from encoder layer as input to decoder
        for dec_layer in range(self.numb_dec_layer):
            # Hidden and cell state inputs
            name = "Decoded_state_h_" + str(dec_layer)
            state_h = Input(shape = [self.lstm_units], name = name)
            inputs.append(state_h)

            name = "Decoded_state_c_" + str(dec_layer)
            state_c = Input(shape = [self.lstm_units], name = name)
            inputs.append(state_c)

            #LSTM layer
            decoder_lstm = LSTM(self.lstm_units, return_sequences = True, name = "Decoder_LSTM_" + str(dec_layer))

            x = decoder_lstm(x, initial_state = [state_h, state_c])

            if self.batch_norm:
                x = BatchNormalization(momentum = self.batch_norm_momentum, name = "BN_decoder_"+str(dec_layer))(x)

        #Dense layer that will return probabilities
        outputs = Dense(self.output_dim, activation = "softmax", name = "Decoder_Dense")(x)

        self.states_to_selfies_model = Model(inputs = inputs, outputs = [outputs], name = "states_to_selfies_model")
        if self.write:
            with open(self.path + 'states_to_selfies.txt', 'w') as f:
                self.states_to_selfies_model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Combine three components
    def build_model(self):
        encoder_inputs = Input(shape = (None,), name = "encoder_inputs")
        decoder_inputs = Input(shape = self.input_shape, name = "decoder_inputs")
        x = self.selfies_to_latent_model(encoder_inputs)
        x = self.latent_to_states_model(x)
        x = [decoder_inputs] + x
        x = self.states_to_selfies_model(x)

        #Full model
        self.model = Model(inputs = [encoder_inputs, decoder_inputs], outputs = [x], name = "Autoencoder")

    # Load autoencoder model from saved weights
    def load_autoencoder_model(self, path):
        self.model.load_weights(path)
        self.build_sample_model()
        self.build_se_to_lat()
    
    # Train
    def fit_model(self, dataX, dataX2, dataY, epochs, batch_size, optimizer):
        self.epochs = epochs
        self.batch_size = batch_size

        if optimizer == 'adam':
            self.optimizer = Adam(learning_rate = 0.001)
        elif optimizer == 'adam_clip':
            self.optimizer = Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False, clipvalue = 3)

        ## Callbacks
        checkpoint_dir = self.path 
        checkpoint_file = (checkpoint_dir + "model--{epoch:02d}.hdf5")
        checkpoint = ModelCheckpoint(checkpoint_file, monitor = "val_loss", mode = "min", save_best_only = True)
        
        #Reduces the learning rate by a factor of 2 when no improvement has been see in the validation set for 2 epochs
        reduce_lr = ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience=2, min_lr = 1e-6)

        #Early Stopping
        early_stop = EarlyStopping(monitor = "val_loss", patience=5)
        
        callbacks_list = [checkpoint, reduce_lr, early_stop]

        self.model.compile(optimizer  = self.optimizer, loss = 'categorical_crossentropy')
       
        results = self.model.fit([dataX, dataX2], dataY, epochs = self.epochs, batch_size =self.batch_size, validation_split=0.1, shuffle = True, verbose = 1, callbacks = callbacks_list)

        fig, ax = plt.subplots()
        ax.plot(results.history['loss'], label = "Train")
        ax.plot(results.history['val_loss'], label = "Val")
        ax.legend()
        ax.set(xlabel='epochs', ylabel = 'loss')
        figure_path = self.path + "Loss_plot.png"
        fig.savefig(figure_path)
        plt.close()

        # Build predictor models
        self.build_sample_model()
        self.build_se_to_lat()

        # Save final models
        self.model.save(self.path + 'AE_model.h5')
        self.sample_model.save(self.path + 'decoder_model.h5')
        self.se_to_lat_model.save(self.path + 'encoder_model.h5')

    # Convert trained autoencoder into latent --> SELFIES model
    def build_sample_model(self):
        # Get the configuration of the batch model
        config = self.states_to_selfies_model.get_config()
        # new_config = config
        # Keep only the "Decoder_Inputs" as single input to the sample_model
        config["input_layers"] = [config["input_layers"][0]]

        # Remove hidden and cell state inputs
        # States will be directly initialized in LSTM cells for prediction
        idx_list = []
        for idx, layer in enumerate(config["layers"]):
            if "Decoded_state_" in layer["name"]:
                idx_list.append(idx)
        for idx in sorted(idx_list, reverse=True):
            config["layers"].pop(idx)

        # Remove inbound_nodes dependencies of remaining layers on deleted ones
        for layer in config["layers"]:
            idx_list = []
            try:
                for idx, inbound_node in enumerate(layer["inbound_nodes"][0]):
                    if "Decoded_state_" in inbound_node[0]:
                        idx_list.append(idx)
            # Catch the exception for first layer (Decoder_Inputs) that has empty list of inbound_nodes[0]
            except:
                pass
            # Pop the inbound_nodes from the list
            # Revert indices to avoid re-arranging
            for idx in sorted(idx_list, reverse=True):
                layer["inbound_nodes"][0].pop(idx)

        # Change the batch_shape of input layer
        config["layers"][0]["config"]["batch_input_shape"] = (
            1,
            1,
            self.output_dim,
        )

        # Finally, change the statefulness of the LSTM layers
        for layer in config["layers"]:
            if "Decoder_LSTM_" in layer["name"]:
                layer["config"]["stateful"] = True

        # Define the sample_model using the modified config file
        sample_model = Model.from_config(config)

        # Copy the trained weights from the trained batch_model to the untrained sample_model
        for layer in sample_model.layers:
            # Get weights from the batch_model
            weights = self.states_to_selfies_model.get_layer(layer.name).get_weights()
            # Set the weights to the sample_model
            sample_model.get_layer(layer.name).set_weights(weights)

        self.sample_model = sample_model
        return config
    
    # Predict latent --> SELFIES
    def latent_to_selfies(self, latent, vocab):
        #predicts the c and h states from the latent representation
        states = self.latent_to_states_model.predict(np.array([latent]))
        
        # Updates the states in the sample model using latent representation
        for dec_layer in range(self.numb_dec_layer): 
            self.sample_model.get_layer("Decoder_LSTM_"+ str(dec_layer)).reset_states(states = [states[2*dec_layer], states[2*dec_layer+1]])
        
        # OHE input
        sample_vector = np.zeros(shape = (1,1,vocab.vocab_size))
        sample_vector[0,0,vocab.char_to_int["G"]] = 1
        selfies = ""
        for i in range(vocab.max_len - 1):
            # Predict character by character, based on previous characters
            pred = self.sample_model.predict(sample_vector)
            idx = np.argmax(pred)
            char = vocab.int_to_char[idx]
            if char == 'G':
                continue
            elif char == 'A':
                break
            else:
                selfies = selfies + char
                sample_vector = np.zeros((1,1,vocab.vocab_size))
                sample_vector[0,0, idx] = 1
        return selfies

    # Convert trained autoencoder into SELFIES --> latent model
    def build_se_to_lat(self):
        # Remove gaussian noise layer
        prediction = self.selfies_to_latent_model.layers[-2].output
        self.se_to_lat_model = Model(inputs = self.selfies_to_latent_model.input, outputs=prediction)

    # Convert SELFIES to latent vectors
    def selfies_to_latentvector(self, vocab, selfies):
        tokens = vocab.tokenize(selfies)
        encoded = np.array(vocab.encode(tokens))
        enc_tensors = []
        for i, en in enumerate(encoded):
            enc_tensors.append(tf.convert_to_tensor(en))
        enc_tensors = tf.convert_to_tensor(enc_tensors)
        latent_vectors = self.se_to_lat_model.predict(enc_tensors)
        return latent_vectors

# Determine percentage of correct molecule reconstruction
def evaluate_reconstruction(real, predicted):
    assert len(real) == len(predicted)
    correct = 0
    for i in range(len(real)):
        if real[i] == predicted[i]:
            correct = correct+1
    return correct/len(real)*100    

# Determine validity of SELFIES predictions
# Output: list of valid SELFIES and percentage of valid SELFIES
def validity(selfies_list):
    total = len(selfies_list)
    valid_selfies = []
    count = 0
    for se in selfies_list:
        sm = sf.decoder(se)
        m = MolFromSmiles(sm)
        if m != None:
            valid_selfies.append(se)
            count += 1
    perc_valid = count/total*100
    
    return valid_selfies, perc_valid


# Train and evaluate the autoencoder
if __name__ == "__main__" :
    run_folder = ''
    file = 'datasets/500k_small_molecule.csv'
    
    # Data preprocessing
    selfies_file = pd.read_csv(file)
    selfies = list(selfies_file['selfies'])
    random.shuffle(selfies)
    vocab = Vocabulary(selfies)
    n_train = int(0.8 * len(selfies))
    selfies_train = selfies[:n_train]
    selfies_test = selfies[n_train:]
    
    tok_train = vocab.tokenize(selfies_train)
    tok_test = vocab.tokenize(selfies_test)
    encode_train = np.array(vocab.encode(tok_train))
    encode_test = vocab.encode(tok_test)
    X_train = vocab.one_hot_encoder(selfies_train)
    Y_train = vocab.get_target(X_train, 'OHE')

    # Model parameters
    latent_dim = 256
    embedding_dim = 256
    lstm_units = 512
    epochs = 200
    batch_size = 128
    batch_norm = True
    batch_norm_momentum = 0.9
    numb_dec_layer = 2
    noise_std = 0.1
    input_shape = X_train.shape[1:]
    output_dim = X_train.shape[-1]

    # Train or load model
    auto = Autoencoder(run_folder, input_shape, latent_dim, lstm_units, output_dim, batch_norm, batch_norm_momentum, noise_std, numb_dec_layer, embedding_dim, vocab.vocab_size, vocab.max_len)
    auto.load_autoencoder_model('models/AE_model.h5')
    #auto.fit_model(encode_train, X_train, Y_train, epochs, batch_size, 'adam')

    # Evaluate trained model
    latent_vectors = auto.se_to_lat_model.predict(encode_test)

    predicted_selfies = []
    for lv in latent_vectors:
        predicted_selfies.append(auto.latent_to_selfies(lv, vocab))

    # Save example predictions to file
    example_predictions = open(run_folder + 'sample_test_predictions.txt', 'w')
    for i in range(len(selfies_test)):
        example_predictions.write('Expected: ' + selfies_test[i])
        example_predictions.write('\nPredicted: ' + predicted_selfies[i] + '\n')
    example_predictions.close()

    # Calculate statistics
    percent_success = evaluate_reconstruction(selfies_test, predicted_selfies)
    print(percent_success)
    _, percent_valid = validity(predicted_selfies)

    test_metrics = open(run_folder + 'results.txt', 'w')
    test_metrics.write('Percent Total Successful: ' + str(round(percent_success, 4)))
    test_metrics.write('\nPercent Valid: ' + str(round(percent_valid, 4)))
    test_metrics.close()

    # Clean-up and remove saved models from training process
    files = os.listdir(run_folder)
    models = [f for f in files if 'model--' in f]
    epochs = [int(m.split('--')[1].split('.')[0]) for m in models]
    max_epoch = max(epochs)
    for e in epochs:
        if e != max_epoch:
            # Single digit epoch --> add 0 before digit
            if int(e/10) == 0:
                file_name = 'model--0' + str(e) + '.hdf5'
            else:
                file_name = 'model--' + str(e) + '.hdf5'
            os.remove(run_folder + file_name)