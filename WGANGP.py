from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import selfies as sf

from time import time
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import utils

from rdkit.Chem import MolFromSmiles, MolToSmiles

class WGANGP():
    def __init__(self, path, input_dim, critic_layers_units, critic_lr, critic_dropout, gp_weight, z_dim, generator_layers_units, generator_batch_norm_momentum, generator_lr, generator_dropout,batch_size, critic_optimizer, gen_optimizer, n_stag_iters):
        self.name = 'WGAN-GP'
        self.path = path
        
        # Model hyperparams
        self.input_dim = input_dim
        self.epoch = 0
        self.batch_size = batch_size

        # Critic hyperparams
        self.critic_layers_units = critic_layers_units
        self.critic_nr_layers = len(critic_layers_units)
        self.critic_lr = critic_lr
        self.critic_dropout = critic_dropout
        self.gp_weight = gp_weight #gradient loss will be weighted by this factor
        
        # Generator hyperparams
        self.z_dim = z_dim  # dimensions of noise fed to generator
        self.generator_layers_units = generator_layers_units
        self.generator_nr_layers = len(generator_layers_units)
        self.generator_batch_norm_momentum = generator_batch_norm_momentum
        self.generator_lr = generator_lr
        self.generator_dropout = generator_dropout

        # Optimizers
        self.optimizer_critic = self.get_optimizer(critic_optimizer, self.critic_lr)
        self.optimizer_generator = self.get_optimizer(gen_optimizer, self.generator_lr)
        
        # Loss values
        self.d_losses_per_gen_iteration = []
        self.g_losses_per_gen_iteration = []
        self.gradient_penalty_loss = []
        self.critic_loss_real = []
        self.critic_loss_fake = []
        self.n_stag_iters = n_stag_iters

        # Create models
        self.build_critic()
        self.build_generator()
        
    # Create optimizer
    def get_optimizer(self, optimizer, lr):
        if optimizer == 'adam':
            # Hyperparameters
            opti = Adam(learning_rate=lr, beta_1 = 0, beta_2 = 0.9)
        return opti
    
    # Create critic model
    # Classifies samples as real or fake
    def build_critic(self):
        critic_input = Input(shape = (self.input_dim,), name = 'critic_input')
        x = critic_input

        # Create dense layers w/ LeakyRelu
        for i in range(self.critic_nr_layers-1):
            x = Dense(self.critic_layers_units[i], name = 'critic_layer_'+str(i))(x)
            # Default: alpha = 0.3 ;  paper: alpha = 0.2
            x = LeakyReLU(alpha = 0.3)(x)
            if self.critic_dropout > 0:
                x = Dropout(self.critic_dropout)(x)
        
        # Make classification in final layer
        critic_output = Dense(1, activation = None, name = 'critic_layer_'+ str(i+1))(x)        
        self.critic = Model(critic_input, critic_output, name = 'Critic')

    # Create generator model
    # Turns noise into fake samples
    def build_generator(self):
        generator_input = Input(shape = (self.z_dim,), name = 'generator_input')
        x = generator_input
        
        # Add layers to generator
        for i in range(self.generator_nr_layers-1):
            x = Dense(self.generator_layers_units[i], name = 'generator_layer_'+str(i))(x)
            if self.generator_batch_norm_momentum:
                x  = BatchNormalization(momentum = self.generator_batch_norm_momentum, name = 'BN_'+str(i))(x)
            if self.generator_dropout > 0:
                x = Dropout(self.generator_dropout)(x)
            # Default: alpha = 0.3 ;  paper: alpha = 0.2
            x = LeakyReLU(alpha = 0.3)(x)
        
        # Output layer
        generator_output = Dense(self.input_dim, activation = None, name = 'generator_layer_'+str(i+1))(x)
        self.generator = Model(generator_input, generator_output, name = 'Generator')
    
    # Train the critic
    # Includes gradient penalty
    def train_critic(self, x_train):
        # Real samples
        data = x_train
        # Noise from normal distribution for generator
        noise = np.random.uniform(-1,1,(self.batch_size, self.z_dim))

        with tf.GradientTape() as critic_tape:
            self.critic.training = True

            # Generate fake samples
            generated_data = self.generator(noise)
            
            # Evaluate samples w/ critic
            real_output = self.critic(data)
            fake_output = self.critic(generated_data)
            
            # Calculate loss
            critic_loss = K.mean(fake_output) - K.mean(real_output)
            self.critic_loss_real.append(K.mean(real_output))
            self.critic_loss_fake.append(K.mean(fake_output))
            
            # Interpolation b/w real and generated data
            # Part of gradient penalty algorithm
            alpha = tf.random.uniform((self.batch_size,1))
            interpolated_samples = alpha*data +(1-alpha)*generated_data
            
            with tf.GradientTape() as t:
                t.watch(interpolated_samples)
                interpolated_samples_output = self.critic(interpolated_samples)
                
            gradients = t.gradient(interpolated_samples_output, [interpolated_samples])
            
            #computing the Euclidean/L2 Norm
            gradients_sqr = K.square(gradients)
            gradients_sqr_sum = K.sum(gradients_sqr, axis = np.arange(1, len(gradients_sqr.shape)))
            gradient_l2_norm = K.sqrt(gradients_sqr_sum)
            gradient_penalty = K.square(1-gradient_l2_norm) #returns the squared distance between L2 norm and 1
            #returns the mean over all the batch samples
            gp =  K.mean(gradient_penalty)
            
            self.gradient_penalty_loss.append(gp)
            # Critic loss
            critic_loss = critic_loss + self.gp_weight * gp
            
        gradients_of_critic = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(gradients_of_critic, self.critic.trainable_variables))
        
        return critic_loss
    
    # Train the generator
    def train_generator(self):
        noise = np.random.normal(0, 1, (self.batch_size, self.z_dim))
        
        with tf.GradientTape() as generator_tape:
            self.generator.training = True
            generated_data = self.generator(noise)
            
            fake_output = self.critic(generated_data)
            
            # Generator loss
            gen_loss = -K.mean(fake_output)
            
        gradients_of_generator = generator_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.optimizer_generator.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        
        return gen_loss
    
    def train(self, x_train, batch_size, epochs, run_folder, autoencoder, vocab, print_every_n_epochs, critic_loops = 5):
        self.n_critic = critic_loops
        self.autoencoder = autoencoder
        self.vocab = vocab

        # Batch and shuffle data
        self.data = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size, drop_remainder = True).shuffle(buffer_size = x_train.shape[0])
        
        train_start = time.time()
        
        # Lists to store loss values
        # Loss_Log: time, epoch, loss
        # Loss: loss
        self.g_loss_log = []
        self.critic_loss_log =[]
        self.critic_loss = []
        self.g_loss = []
            
        for epoch in range (self.epoch, self.epoch+epochs):
            critic_loss_per_batch = []
            g_loss_per_batch = []
            batches_done = 0
            
            for i, batch in enumerate(self.data):
                # Train the critic
                # Trained every batch iteration
                loss_d = self.train_critic(batch)
                critic_loss_per_batch.append(loss_d)
                
                # Train the Generator
                # Trained n_critic batch iterations
                if i % self.n_critic == 0:
                    loss_g = self.train_generator()
                    g_loss_per_batch.append(loss_g)
                    batches_done = batches_done +  self.n_critic
                
                # Save information if it is the last batch ---> end of an epoch
                if i == len(self.data) -1:
                    # Calculate losses for this epoch, based on batch losses
                    self.critic_loss_log.append([time.time()-train_start, epoch, np.mean(critic_loss_per_batch)])
                    self.g_loss_log.append([time.time()-train_start, epoch, np.mean(g_loss_per_batch)])
                    self.critic_loss.append(np.mean(critic_loss_per_batch))
                    self.g_loss.append(np.mean(g_loss_per_batch))		   
                    print( 'Epochs {}: D_loss = {}, G_loss = {}'.format(epoch, self.critic_loss_log[-1][2], self.g_loss_log[-1][2]))

                    # Save information
                    if (epoch % print_every_n_epochs) == 0:
                        print('Saving...')
                        # Save general model information
                        self.save_model(run_folder)
                        self.plot_loss(run_folder)
                        self.plot_gp_loss(run_folder)
                        # Save current epoch information
                        self.critic.save_weights(os.path.join(run_folder, 'weights/critic_weights.h5'))
                        self.generator.save_weights(os.path.join(run_folder, 'weights/generator_weights.h5'))
                        self.sample_data(200, run_folder, save=True)
                    if epoch == 2500 or epoch == 5000 or epoch == 7500:
                        self.plot_loss(run_folder)
                        self.critic.save_weights(os.path.join(run_folder, 'weights/critic_weights_' + str(epoch) + '.h5'))
                        self.generator.save_weights(os.path.join(run_folder, 'weights/generator_weights_' + str(epoch) + '.h5'))
            self.epoch += 1

    def sample_data(self, n, run_folder, save):
        print('Sampling data...')

        # Make generations
        noise = np.random.uniform(-1,1,(n, self.z_dim))
        generated_data = self.generator.predict(noise)

        # Transform generations into SELFIES with decoder
        generated_selfies = []
        for i in range(generated_data.shape[0]):
            sml = self.autoencoder.latent_to_selfies(generated_data[i], self.vocab)
            generated_selfies.append(sml)
        
        valid_selfies, perc_valid = validity(generated_selfies)
        if save == True: 
            f = open(os.path.join(run_folder, "Generations/samples_epoch_%d.txt" % (self.epoch)), 'w')
            f.write('Percent Valid: ' + str(perc_valid))
            for sm in generated_selfies:
                f.write('\n' + sm)
            f.close()

        return valid_selfies

    # Save generator and critic models
    def save_model(self, run_folder):
        self.critic.save(os.path.join(run_folder, 'critic.h5'))
        self.generator.save(os.path.join(run_folder, 'generator.h5'))

    # Load weights from saved generator and critic models
    def load_weights(self, filepath_critic, filepath_generator, vocab, autoencoder):
        self.critic.load_weights(filepath_critic)
        self.generator.load_weights(filepath_generator)
        self.vocab = vocab
        self.autoencoder = autoencoder
    
    # Plot training loss of generator and critic models
    def plot_loss(self, run_folder):
        # Clean out loss values to remove outliers that skew the graph
        g_loss_cleaned = []
        c_loss_cleaned = []
        for i in range(len(self.g_loss)):
            if self.g_loss[i] < 10 and self.g_loss[i] > -10 and self.critic_loss[i] < 10 and self.critic_loss[i] > -10:
                g_loss_cleaned.append(self.g_loss[i])
                c_loss_cleaned.append(self.critic_loss[i])

        # Plot regular
        fig, ax = plt.subplots()
        ax.plot(self.g_loss, label = "G_loss")
        ax.plot(self.critic_loss, label = "D_loss")
        ax.legend()
        ax.set(xlabel='Epoch', ylabel = 'Loss')
        if self.epoch == 2500:
            figure_path = os.path.join(run_folder, 'viz/loss_plot_2500.png')
        elif self.epoch == 5000:
            figure_path = os.path.join(run_folder, 'viz/loss_plot_5000.png')
        elif self.epoch == 7500:
            figure_path = os.path.join(run_folder, 'viz/loss_plot_7500.png')
        else:
            figure_path = os.path.join(run_folder, 'viz/loss_plot.png')
        fig.savefig(figure_path)
        plt.close()

        # Plot cleaned
        fig, ax = plt.subplots()
        ax.plot(g_loss_cleaned, label = "G_loss")
        ax.plot(c_loss_cleaned, label = "D_loss")
        ax.legend()
        ax.set(xlabel='Epoch', ylabel = 'Loss')
        if self.epoch == 2500:
            figure_path = os.path.join(run_folder, 'viz/loss_plot_2500_cleaned.png')
        elif self.epoch == 5000:
            figure_path = os.path.join(run_folder, 'viz/loss_plot_5000_cleaned.png')
        elif self.epoch == 7500:
            figure_path = os.path.join(run_folder, 'viz/loss_plot_7500_cleaned.png')
        else:
            figure_path = os.path.join(run_folder, 'viz/loss_plot_cleaned.png')
        fig.savefig(figure_path)
        plt.close()
    
    # Plot training loss for gradient penalty
    def plot_gp_loss(self, run_folder):
        plt.plot(self.gradient_penalty_loss)
        plt.title('Gradient Penalty Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        figure_path = os.path.join(run_folder, 'viz/gp_loss_plot.png')
        plt.savefig(figure_path)
        plt.close()

    # Plot architecture of generator and critic models
    def plot_model(self, run_folder):
        plot_model(self.critic, to_file=os.path.join(run_folder, 'viz/critic.png', show_shapes = True, show_layer_names = True))
        plot_model(self.generator, to_file=os.path.join(run_folder, 'viz/generator.png', show_shapes = True, show_layer_names = True))

    # Generate compounds from trained WGAN-GP
    def make_generations(self, n, save_path, checkpoint_path, checkpoints):
        print('Started sampling...')

        # Make generations
        noise = np.random.uniform(-1,1,(n, self.z_dim))
        generated_data = self.generator.predict(noise)

        # Transform generations into SELFIES and SMILES with decoder
        generated_selfies = []
        canon_smiles = []
        for i in range(generated_data.shape[0]):
            sel = self.autoencoder.latent_to_selfies(generated_data[i], self.vocab)
            sm = sf.decoder(sel)
            if sm != None:
                mol = MolFromSmiles(sm)
                # Check that molecule is valid before adding
                if mol != None:
                    generated_selfies.append(sel)
                    canon_smiles.append(MolToSmiles(mol))
            
            # Checkpoint
            if (i % 1000) == 0 and i != 0:
                if i in checkpoints:
                    df = pd.DataFrame()
                    df['canonical_smiles'] = canon_smiles
                    df['selfies'] = generated_selfies
                    df.to_csv(checkpoint_path + 'gen_' + str(i) + '.csv', index=False)
                else:
                    df = pd.DataFrame()
                    df['canonical_smiles'] = canon_smiles
                    df['selfies'] = generated_selfies
                    df.to_csv(checkpoint_path + 'checkpoints.csv', index=False)
            
            print(str(i) + ' Done')

        df = pd.DataFrame()
        df['canonical_smiles'] = canon_smiles
        df['selfies'] = generated_selfies
        n_valid = len(df)
        df.drop_duplicates(subset='canonical_smiles', inplace=True)
        df.to_csv(save_path, index=False)

        print('Validity: ' + str(n_valid / n))
        print('Uniqueness: ' + str(len(df) / n_valid) + '%')

# Determine validity of SELFIES generations
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

# Initialize the WGAN-GP model
def initialize(main_path):
    # Create GAN
    input_dim = 256
    critic_layers_units = [256,256,256]
    critic_lr = 0.0001
    gp_weight = 10
    z_dim  = 64
    generator_layers_units = [128,256,256,256,256]
    generator_batch_norm_momentum = 0.9
    generator_lr = 0.0001
    batch_size = 64
    critic_optimizer = 'adam'
    generator_optimizer = 'adam'
    critic_dropout = 0.2
    generator_dropout = 0.2
    n_stag_iters = 50

    gan = WGANGP(main_path, input_dim, critic_layers_units, critic_lr, critic_dropout, gp_weight, z_dim, generator_layers_units, generator_batch_norm_momentum, generator_lr, generator_dropout,batch_size, critic_optimizer, generator_optimizer, n_stag_iters)
    return gan

# Sample generations from WGAN-GP
def sample(gan, gen_path, critic_path, vocab, auto, n, save_path, checkpoint_path, checkpoints):
    gan.load_weights(critic_path, gen_path, vocab, auto)
    gan.make_generations(n, save_path, checkpoint_path, checkpoints)


# Compound generation
if __name__ == '__main__':
    vocab_path = 'datasets/500k_small_molecule.csv'
    ae_path = 'models/AE_model.h5'
    critic_path = 'models/pretrained_critic.h5'
    gen_path = 'models/TL_generator.h5'

    run_folder = ''
    save_path = 'generations.csv'
    checkpoint_folder = ''

    vocab, auto = utils.initialize_models(run_folder, vocab_path, ae_path)

    gan = initialize(run_folder)
    gan.load_weights(critic_path, gen_path, vocab, auto)
    sample(gan, gen_path, critic_path, vocab, auto, 500000, save_path, checkpoint_folder, [100000, 200000, 300000, 400000])

