# BACE1-Drug-Discovery
The training datasets, python scripts, trained models, and generated BACEE1 candidate inhibitors from the *de novo* generative AI drug discovery framework developed in "An AI-Driven Framework to Discover BACE1 Inhibitors for Alzheimer’s Disease."

* `datasets/`: contains the preprocessed datasets
* `models/`: contains the trained weights for the autoencoder, QSAR-NNs, pre-trained WGAN-GP, TL model, and GA model
* `docking/`: contains the prepared BACE1/BACE2 crystal structures and the BACE1/BACE2 active sites
* `generated compounds/`: contains the SMILES, SELFIES, and molecular property values of filtered compounds
* `candidate inhibitors/`: contains the binding poses to the BACE1 active site of the discovered candidate inhibitors
* `preprocess.py`: preprocesses the datasets
* `Vocabulary.py`: processes SELFIES strings for the autoencoder model
* `AE.py`: trains and runs the autoencoder model
* `predictor_lv.py`: trains and runs the QSAR-NNs for pIC50, logBB, log(Cmax), and log(t1/2) prediction
* `WGANGP.py`: implementation of the Wasserstein GAN with Gradient Penalty (WGAN-GP)
* `TL.py`: implementation of the transfer learning model
* `GA.py`: implementation of the Genetic Algorithm
* `run_WGANGP.py`: trains and runs the WGAN-GP
* `run_TL.py`: trains and runs the TL model
* `run_GA.py`: trains and runs the GA
* `docking_prep`: prepares compounds and proteins for AutoDock Vina molecular docking simulation
* `docking_analysis`: analyzes molecular docking results, based on binding energy and selectivity score, to identify candidate inhibitors
* `visualize.py`: various visualization tools for the framework
* `utils.py`: various utility methods for the framework
* `requirements.txt`: the package requirements for running the framework in python 3.10