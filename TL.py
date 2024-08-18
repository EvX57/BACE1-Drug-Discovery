from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from time import time
import os
from scipy.stats import wasserstein_distance

import WGANGP
from WGANGP import WGANGP
import utils

class TL(WGANGP):
    def __init__(self, path, input_dim, critic_layers_units, critic_lr, critic_dropout, gp_weight, z_dim, generator_layers_units, generator_batch_norm_momentum, generator_lr, generator_dropout,batch_size, critic_optimizer, gen_optimizer, n_stag_iters, critic_transfer_layers, gen_transfer_layers, critic_path, gen_path, vocab, auto):
        super().__init__(path, input_dim, critic_layers_units, critic_lr, critic_dropout, gp_weight, z_dim, generator_layers_units, generator_batch_norm_momentum, generator_lr, generator_dropout,batch_size, critic_optimizer, gen_optimizer, n_stag_iters)
        self.name = 'Transfer Learning WGAN-GP'
        
        # Params
        self.vocab = vocab
        self.autoencoder = auto

        # Critic hyperparams
        self.critic_transfer_layers = critic_transfer_layers
        self.critic_path = critic_path

        # Generator hyperparams
        self.gen_transfer_layers = gen_transfer_layers
        self.gen_path = gen_path

        # Load weights from pre-trained WGAN-GP
        self.load_weights(self.critic_path, self.gen_path, self.vocab, self.autoencoder)

        # Create models
        #self.build_critic_transfer(True)
        #self.build_generator_transfer(True)
        self.build_critic_transfer_v2(True)
        self.build_generator_transfer_v2(True)
        
    # Create critic model
    # Classifies samples as real or fake
    # The transfer layers are created from scratch
    def build_critic_transfer(self, print=False):
        # Prepare the WGAN-GP critic
        num_layers = len(self.critic.layers)
        # Get the input
        critic_input = self.critic.input
        # Locate the index to truncate the model
        count = 0
        index = num_layers - 1
        while count < (self.critic_transfer_layers + 1):
            if index < 0:
                print("ERROR")
            if 'critic_layer' in self.critic.layers[index].name:
                count += 1
            index -= 1
        # Set the layers to untrainable
        for i in range(index + 1):
            self.critic.layers[i].trainable = False
        # Truncate the layers at the end
        x = self.critic.layers[index].output

        # Create new layers of the transfer model
        for i in range(self.critic_transfer_layers):
            x = Dense(self.critic_layers_units[i], name = 'transfer_critic_layer_'+str(i))(x)
            # Default: alpha = 0.3 ;  paper: alpha = 0.2
            x = LeakyReLU(alpha = 0.3)(x)
            if self.critic_dropout > 0:
                x = Dropout(self.critic_dropout)(x)
        
        # Make classification in final layer
        critic_output = Dense(1, activation = None, name = 'transfer_critic_layer_'+ str(i+1))(x)        
        self.critic = Model(critic_input, critic_output, name = 'Critic')

        # Print model architecture
        if print:
            with open(self.path + 'transfer_critic_summary.txt', 'w') as f:
                self.critic.summary(print_fn=lambda x: f.write(x + '\n'))

    # Create generator model
    # Turns noise into fake samples
    # The transfer layers are created from scratch
    def build_generator_transfer(self, print=False):
        # Prepare the WGAN-GP generator
        num_layers = len(self.generator.layers)
        # Get the input
        generator_input = self.generator.input
        # Locate the index to truncate the model
        count = 0
        index = num_layers - 1
        while count < (self.gen_transfer_layers + 1):
            if index < 0:
                print("ERROR")
            if 'generator_layer' in self.generator.layers[index].name:
                count += 1
            index -= 1
        # Set the layers to untrainable
        for i in range(index + 1):
            self.generator.layers[i].trainable = False
        # Truncate the layers at the end
        x = self.generator.layers[index].output

        # Add transfer layers to generator
        for i in range(self.gen_transfer_layers):
            index = i + self.generator_nr_layers - self.gen_transfer_layers - 1
            x = Dense(self.generator_layers_units[index], name = 'transfer_generator_layer_'+str(i))(x)
            if self.generator_batch_norm_momentum:
                x  = BatchNormalization(momentum = self.generator_batch_norm_momentum)(x)
            if self.generator_dropout > 0:
                x = Dropout(self.generator_dropout)(x)
            # Default: alpha = 0.3 ;  paper: alpha = 0.2
            x = LeakyReLU(alpha = 0.3)(x)
        
        # Output layer
        generator_output = Dense(self.input_dim, activation = None, name = 'transfer_generator_layer_'+str(i+1))(x)
        self.generator = Model(generator_input, generator_output, name = 'Generator')

        # Print model architecture
        if print:
            with open(self.path + 'transfer_generator_summary.txt', 'w') as f:
                self.generator.summary(print_fn=lambda x: f.write(x + '\n'))       
    
    # Create critic model
    # Classifies samples as real or fake
    # The transfer layers are not created from scratch
    def build_critic_transfer_v2(self, print_model=False):
        # Get the number of layers
        num_layers = len(self.critic.layers)

        # Locate the index to truncate the model
        count = 0
        index = num_layers - 1
        while count < (self.critic_transfer_layers + 1):
            if index < 0:
                print("ERROR")
            if 'critic_layer' in self.critic.layers[index].name:
                count += 1
            index -= 1

        # Set the layers to untrainable and trainable
        # Rename the trainable layers to transfer layers
        t_count = 1
        #config = self.critic.get_config()
        for i in range(num_layers):
            if i <= index:
                self.critic.layers[i].trainable = False
            else:
                self.critic.layers[i].trainable = True
                #self.critic.layers[i]._name = 'transfer_layer_' + str(t_count)
                '''if 'critic_layer_' in self.critic.layers[i].name:
                    self.critic.layers[i]._name = 'transfer_layer_' + str(t_count)  # FIX
                    #config['layers'][i]['config']['name'] = 'transfer_layer_' + str(t_count)'''
                t_count += 1

        # Print model architecture
        if print_model:
            with open(self.path + 'transfer_critic_summary.txt', 'w') as f:
                self.critic.summary(print_fn=lambda x: f.write(x + '\n'))
        
    # Create generator model
    # Turns noise into fake samples
    # The transfer layers are not created from scratch
    def build_generator_transfer_v2(self, print=False):
        # Get the number of layers
        num_layers = len(self.generator.layers)
        
        # Locate the index to truncate the model
        count = 0
        index = num_layers - 1
        while count < (self.gen_transfer_layers + 1):
            if index < 0:
                print("ERROR")
            if 'generator_layer' in self.generator.layers[index].name:
                count += 1
            index -= 1
        
        # Set the layers to untrainable and trainable
        # Rename the trainable layers to transfer layers
        t_count = 1
        for i in range(num_layers):
            if i <= index:
                self.generator.layers[i].trainable = False
            else:
                self.generator.layers[i].trainable = True
                #self.generator.layers[i]._name = 'transfer_layer_' + str(t_count)
                '''if 'generator_layer_' in self.generator.layers[i].name:
                    self.generator.layers[i]._name = 'transfer_layer_' + str(t_count)
                    t_count += 1'''
                t_count += 1

        # Print model architecture
        if print:
            with open(self.path + 'transfer_generator_summary.txt', 'w') as f:
                self.generator.summary(print_fn=lambda x: f.write(x + '\n'))  
  
    # Save generator and critic models
    def save_model(self, run_folder):
        self.critic.save(os.path.join(run_folder, 'critic.h5'))
        self.generator.save(os.path.join(run_folder, 'generator.h5'))

# Compare TL models based on critic outputs of trained BACE1 and 100k WGANGPs
# Calculate loss_1 between generated samples and BACE1 dataset on BACE1 critic
# Calculate loss_2 between generated samples and 100k dataset on 100k critic
def compare_critic_distribution(dfs, loss_functions):
    # Model paths
    main_path = ''
    vocab_path = 'datasets/500k_small_molecule.csv'
    ae_path = 'models/AE_model.h5'
    bace1_critic = ''
    bace1_gen = ''
    chembl_critic = 'models/pretrained_critic.h5'
    chembl_gen = 'models/pretrained_generator.h5'
   
    # Load data
    bace1_df = pd.read_csv('datasets/BACE1.csv')
    chembl_df = pd.read_csv('datasets/100k_small_molecule.csv')

    # Load autoencoder
    vocab, auto = utils.initialize_models(main_path, vocab_path, ae_path)

    # Load WGANGP
    bace1_gan = WGANGP.initialize(main_path)
    bace1_gan.load_weights(bace1_critic, bace1_gen, vocab, auto)
    chembl_gan = WGANGP.initialize(main_path)
    chembl_gan.load_weights(chembl_critic, chembl_gen, vocab, auto)

    # Convert molecules to latent vectors
    lat_vecs = []
    for df in dfs:
        lv = calculate_lv(df, vocab, auto)
        lat_vecs.append(lv)

    bace1_lv = calculate_lv(bace1_df, vocab, auto)
    chembl_lv = calculate_lv(chembl_df, vocab, auto)
    

    # Calculate loss values from latent vectors
    loss_function_vals = []
    for loss_func in loss_functions:
        loss_vals = []
        for lv in lat_vecs:
            loss_bace1 = loss_func(bace1_gan, lv, bace1_lv)
            loss_chembl = loss_func(chembl_gan, lv, chembl_lv)
            loss = math.sqrt(math.pow(loss_bace1, 2) + math.pow(loss_chembl, 2))
            loss_vals.append(loss)

            print('BACE1 Loss: ' + str(loss_bace1))
            print('Chembl Loss: ' + str(loss_chembl) + '\n')
        loss_function_vals.append(loss_vals)

    # Return loss
    return loss_function_vals

# Calculate and return latent vectors
def calculate_lv(df, vocab, auto):
    selfies = list(df['selfies'])
    tok = vocab.tokenize(selfies)
    encoded = vocab.encode(tok)
    lv = auto.se_to_lat_model.predict(encoded)
    return lv

# Create bins of distribution
def create_bins(a, b, bins, viz=False):
    # counter variable has to be initialized outside method
    global counter

    # Find range
    range_min = min(min(a), min(b))
    range_max = max(max(a), max(b))
    range = [range_min, range_max]

    # Create bins
    hist_a, _ = np.histogram(a, bins, range, False)
    hist_b, _ = np.histogram(b, bins, range, False)

    # Remove 0's
    hist_a = hist_a + 1e-10
    hist_b = hist_b + 1e-10

    # Scale
    hist_a = hist_a / hist_a.sum()
    hist_b = hist_b / hist_b.sum()

    # Convert
    hist_a = hist_a.tolist()
    hist_b = hist_b.tolist()

    # Visualize
    if viz:
        plt.hist(a, bins, range)
        plt.savefig('hist_bins' + str(bins) + '_' + str(counter) + '.png')
        counter += 1
        plt.close()
        plt.hist(b, bins, range)
        plt.savefig('hist_bins' + str(bins) + '_' + str(counter) + '.png')
        counter += 1
        plt.close()

    return hist_a, hist_b

# Calculate Earth Mover's Distance between two distributions
def loss_EM(gan, lv_sample, lv_reference, n_bins = 100):
    # Predict values
    sample = gan.critic.predict(lv_sample)
    ref = gan.critic.predict(lv_reference)
    sample = [x[0] for x in sample if x[0] >= -3.0 and x[0] <= 3.0]
    ref = [x[0] for x in ref if x[0] >= -3.0 and x[0] <= 3.0]

    # Create bins
    hist_sample, hist_ref = create_bins(sample, ref, n_bins)

    # Calculate EM distance
    return wasserstein_distance(hist_sample, hist_ref)

# Compare between EMD losses of different TL models
def distribution_ranking_order(folder_path, loss_functions, lf_names, save_path_prefix):
    dfs = []
    names = []

    files = os.listdir(folder_path)
    folders = [f for f in files if 'TL' in f or 'OG' in f]
    for f in folders:
        path = folder_path + f + '/10k Sample/10k_gen.csv'
        dfs.append(pd.read_csv(path))
        names.append(f)

    loss_function_vals = compare_critic_distribution(dfs, loss_functions)

    for i, loss_vals in enumerate(loss_function_vals):
        df = pd.DataFrame()
        df['Model Name'] = names
        df[lf_names[i]] = loss_vals
        df.sort_values(by=lf_names[i], inplace=True)
        df.to_csv(save_path_prefix + lf_names[i] + '_ranking.csv', index=False)