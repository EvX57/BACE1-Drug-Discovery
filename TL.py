from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

from time import time
import os

from WGANGP import WGANGP

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