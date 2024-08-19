import tensorflow as tf
import statistics
import selfies as sf
from rdkit.Chem import MolFromSmiles, MolToSmiles

from time import time
import numpy as np
import pandas as pd
import os
import time

import matplotlib.pyplot as plt

from TL import TL
import utils
import visualize

class GA(TL):
    def __init__(self, path, input_dim, critic_layers_units, critic_lr, critic_dropout, gp_weight, z_dim, generator_layers_units, generator_batch_norm_momentum, generator_lr, generator_dropout, batch_size, critic_optimizer, gen_optimizer, n_stag_iters, critic_transfer_layers, gen_transfer_layers, critic_path, gen_path, vocab, auto, predictors):
        super().__init__(path, input_dim, critic_layers_units, critic_lr, critic_dropout, gp_weight, z_dim, generator_layers_units, generator_batch_norm_momentum, generator_lr, generator_dropout, batch_size, critic_optimizer, gen_optimizer, n_stag_iters, critic_transfer_layers, gen_transfer_layers, critic_path, gen_path, vocab, auto)
        
        self.predictors = predictors

        self.replacement_start_epoch = 0
        self.replacement_percent = 0.1
        self.current_samples = []

        # Store the model iteration that produces the most fit samples
        self.best_avg_fitness = None
        self.best_epoch = None
        self.best_critic = None
        self.best_gen = None

    # Calculate fitness based on predicted pIC50, logBB, logCmax, and logThalf of compounds
    def calculate_fitness(self, samples, lv=True):
        if lv:
            # Calculate metrics
            samples = np.array(samples)
            p = self.predictors['pIC50'].predict(samples, string=False)
            b = self.predictors['logBB'].predict(samples, string=False)
            c = self.predictors['logCmax'].predict(samples, string=False)
            t = self.predictors['logThalf'].predict(samples, string=False)
        else:
            p = samples[0]
            b = samples[1]
            c = samples[2]
            t = samples[3]
        
        fitness_vals = []
        for i in range(len(p)):
            fitness_vals.append(p[i] + b[i] + c[i] + t[i])

        return fitness_vals

    # Model traning with different selection algorithms
    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_epochs, critic_loops=5, replace_every_n_epochs=100):        
        self.n_critic = critic_loops
        self.replace_every_n_epochs = replace_every_n_epochs

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
        self.train_fitness = []
        self.gen_fitness_avg = []
        self.gen_fitness_max = []
        self.gen_fitness_min = []

        for epoch in range(self.epoch, self.epoch+epochs):
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
                    if (epoch % print_every_n_epochs) == 0 and epoch != 0:
                        print('Saving...')
                        # Save general model information
                        self.save_model(run_folder)
                        self.plot_loss(run_folder)
                        # Save current epoch information
                        self.critic.save_weights(os.path.join(run_folder, 'weights/critic_weights.h5'))
                        self.generator.save_weights(os.path.join(run_folder, 'weights/generator_weights.h5'))
                        self.sample_data(200, run_folder, save=True)
                    
                    # Perform genetic operations
                    # Input is selfies representations of molecules
                    if (epoch % replace_every_n_epochs) == 0 and epoch >= self.replacement_start_epoch:
                        # Generate New Samples for Replacement
                        noise = np.random.uniform(-1,1,(len(x_train), self.z_dim))
                        generated_data = self.generator.predict(noise)
                        generated_samples = generated_data.tolist()

                        # Replacement
                        x_train = self.replace_elitism(x_train, generated_samples, run_folder)

                        # Batch and shuffle data
                        self.data = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size, drop_remainder = True).shuffle(buffer_size = x_train.shape[0])

                        # Save new training distribution
                        if (epoch % 50) == 0:
                            df_train_samples = pd.DataFrame()
                            df_train_samples['LV'] = x_train.tolist()
                            df_train_samples.to_csv(run_folder + 'train_sample_lv.csv', index=False)

                        # Check if this is the most fit model iteration
                        # Save if so
                        if self.best_avg_fitness == None or self.gen_fitness_avg[-1] > self.best_avg_fitness:
                            self.best_avg_fitness = self.gen_fitness_avg[-1]
                            self.best_epoch = self.epoch
                            self.critic.save_weights(os.path.join(run_folder, 'weights/fittest_critic_weights_e' + str(epoch) + '.h5'))
                            self.generator.save_weights(os.path.join(run_folder, 'weights/fittest_generator_weights_e' + str(epoch) + '.h5'))

                        # Reset Adam Optimizer
                        print('Resetting optimizer...')
                        self.optimizer_critic = self.get_optimizer('adam', self.critic_lr)
                        self.optimizer_generator = self.get_optimizer('adam', self.generator_lr)

            self.epoch += 1
        
        f = open(run_folder + 'weights/fittest_epoch.txt', 'w')
        f.write('Fittest Epoch: ' + str(self.best_epoch))
        f.close()

    # Calculate the average fitness of a set of samples
    def calculate_average_fitness(self, values):
        threshold = 0.0

        sum = 0
        count = 0
        invalid = 0
        for v in values:
            if v > threshold:
                sum += v
                count += 1
            else:
                invalid += 1
        
        print("# Invalid: " + str(invalid))
        return sum / count
            
    # Take best generated samples and replace with worst
    # Ranking determined by fitness score
    def replace_elitism(self, x_train, generated_samples, run_folder):
        n_replacements = int(len(x_train) * self.replacement_percent)
        n_kept = len(x_train) - n_replacements

        # Highest fitness first, lowest fitness last
        generated_fitness = self.calculate_fitness(generated_samples)
        self.gen_fitness_avg.append(self.calculate_average_fitness(generated_fitness))
        self.gen_fitness_max.append(max(generated_fitness))
        self.gen_fitness_min.append(min(generated_fitness))
        sorted_gen_samples = [val for (_, val) in sorted(zip(generated_fitness, generated_samples), key=lambda x:x[0], reverse=True)]
        best_gen_samples = sorted_gen_samples[:n_replacements]
        best_gen_samples = [np.asarray(s, dtype='float32') for s in best_gen_samples]

        # Find best real samples
        # Highest fitness first, lowest fitness last
        train_fitness = self.calculate_fitness(x_train)
        self.train_fitness.append(statistics.mean(train_fitness))
        sorted_train_samples = [val for (_, val) in sorted(zip(train_fitness, x_train), key=lambda x:x[0], reverse=True)]
        best_train_samples = sorted_train_samples[:n_kept]

        # Combine
        best_train_samples.extend(best_gen_samples)

        # Plot
        self.plot_fitness_progression(run_folder)

        return np.array(best_train_samples, dtype='float32')

    # Plot the change in fitness over epochs
    def plot_fitness_progression(self, run_folder):
        epochs = [(self.replacement_start_epoch + i*self.replace_every_n_epochs) for i in range(len(self.train_fitness))]
        
        # Train samples
        plt.plot(epochs, self.train_fitness)
        plt.xlabel('Epoch')
        plt.ylabel('Fitness')
        plt.title('Train Fitness Progression')
        plt.savefig(run_folder + 'train_fitness_progression.png')
        plt.close()

        # Generated Samples
        # Avg
        plt.plot(epochs, self.gen_fitness_avg)
        plt.xlabel('Epoch')
        plt.ylabel('Fitness')
        plt.title('Gen Fitness Progression (Avg)')
        plt.savefig(run_folder + 'gen_fitness_progression_avg.png')
        plt.close()
        # Max
        plt.plot(epochs, self.gen_fitness_max)
        plt.xlabel('Epoch')
        plt.ylabel('Fitness')
        plt.title('Gen Fitness Progression (Max)')
        plt.savefig(run_folder + 'gen_fitness_progression_max.png')
        plt.close()

# Visualize improvement in fitness during GA training
# folder: folder of generated samples every 250 training epochs
# extract: whether to extract generations or retrieve from saved .csv file
def visualize_fitness_progression(folder, extract=True):
    paths = os.listdir(folder)
    epochs = []
    distributions = []
    avgs = []

    # Extract generations, calculate fitness, save in .csv
    if extract:
        paths = [p for p in paths if '.txt' in p]
        epochs = [int(p.split('.')[0].split('_')[-1]) for p in paths]
        epochs.sort()

        for e in epochs:
            name = 'samples_epoch_' + str(e)
            file = open(folder + name + '.txt')

            # Get selfies
            line = file.readline()
            line = file.readline()
            selfies = []
            while line != '':
                s = line.split('\n')[0]
                if s != '':
                    selfies.append(s)
                line = file.readline()

            # Remove duplicates
            selfies = list(set(selfies))

            # Calculate canonical smiles
            smiles = []
            valid_selfies = []
            for sel in selfies:
                sm = sf.decoder(sel)
                if sm != None:
                    mol = MolFromSmiles(sm)
                    # Check that molecule is valid before adding
                    if mol != None:
                        valid_selfies.append(sel)
                        smiles.append(MolToSmiles(mol))

            # Save to df
            df_path = folder + name + '.csv'
            df = pd.DataFrame()
            df['canonical_smiles'] = smiles
            df['selfies'] = valid_selfies
            df.to_csv(df_path, index=False)

            # Calculate metrics
            utils.calculate_metrics(df_path, df_path, metrics=['pIC50', 'logBB', 'logCmax', 'logThalf'])

            # Calculate fitness
            utils.calculate_fitness(df_path)
            df = pd.read_csv(df_path)
            fitness = list(df['Fitness'])

            # Store values
            # Remove outliers
            distributions.append([f for f in fitness if f > 0.0])
            avgs.append(statistics.mean([f for f in fitness if f > 0.0]))

            print('e' + str(e) + ' Done')
    # Retrieve generations and their fitness values from saved .csv files
    else:
        paths = [p for p in paths if '.csv' in p]
        epochs = [int(p.split('.')[0].split('_')[-1]) for p in paths]
        epochs.sort()

        for e in epochs:
            # Retrieve values
            name = 'samples_epoch_' + str(e)
            df = pd.read_csv(folder + name + '.csv')
            fitness = list(df['Fitness'])

            # Store values
            # Remove outliers
            distributions.append([f for f in fitness if f > 0.0])
            avgs.append(statistics.mean([f for f in fitness if f > 0.0]))

    # Plot
    plt.plot(epochs, avgs)
    plt.title('Average Fitness vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Fitness')
    plt.tight_layout()
    plt.savefig(folder + 'avg_fitness.png')
    plt.close()

    indices = [0, int(len(epochs)/4), int(len(epochs)/2), int(3*len(epochs)/4), -1]
    d = [distributions[i] for i in indices]
    names = ['e' + str(epochs[i]) for i in indices]
    visualize.compare_property_distribution(d, names, 'Fitness', 'Fitness Score Distribution', folder + 'fitness_distribution.png')