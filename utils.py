import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, RDConfig, FilterCatalog
from rdkit.Chem import MolFromSmiles, MolToSmiles
import selfies as sf

from AE import Autoencoder as AE
from Vocabulary import Vocabulary
import predictor_lv

import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

# Check SMILES --> SELFIES --> SMILES conversion
def SELFIES_to_SMILES_check(df):
    fail_counter = 0
    for i in range(len(df)):
        dec_sm = sf.decoder(df.at[i, 'selfies'])
        dec_sm = Chem.CanonSmiles(dec_sm)
        sm = Chem.CanonSmiles(df.at[i, 'canonical_smiles'])
        if dec_sm != sm:
            print("FAIL")
            fail_counter += 1
    print(fail_counter)

# Convert from SELFIES to SMILES in a dataframe
def SELFIES_to_SMILES(df_path, save_path):
    df = pd.read_csv(df_path)
    selfies = list(df['selfies'])

    indices = []
    smiles = []
    for i, se in enumerate(selfies):
        try:
            sm = sf.decoder(se)
            if sm != None:
                mol = MolFromSmiles(sm)
                # Check that molecule is valid before adding
                if mol != None:
                    indices.append(i)
                    smiles.append(MolToSmiles(mol))
        except AttributeError:
            continue
    
    df = df.loc[indices]
    df['canonical_smiles'] = smiles
    df.to_csv(save_path, index=False)

# Calculate MW from SELFIES
def calculate_MW(selfies):
    smiles = [sf.decoder(s) for s in selfies]
    mols = [Chem.MolFromSmiles(sm) for sm in smiles]
    mws = [Descriptors.MolWt(m) for m in mols if m != None]
    return mws

# Calculate QED from SELFIES
def calculate_qed(selfies):
    smiles = [sf.decoder(s) for s in selfies]
    mols = [Chem.MolFromSmiles(sm) for sm in smiles]
    qeds = [Chem.QED.default(m) for m in mols if m != None]
    return qeds

# Calculate SAS from SELFIES
def calculate_sas(selfies):
    smiles = [sf.decoder(s) for s in selfies]
    mols = [Chem.MolFromSmiles(sm) for sm in smiles]

    sas = []
    for m in mols:
        try:
            sas.append(sascorer.calculateScore(m))
        except ZeroDivisionError:
            #continue
            sas.append(9.0)
    
    return sas

# Calculate the number of PAINS substructures in a compound (input as SMILES)
def n_PAINS(smiles):
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)

    n_pains = []
    for i, sm in enumerate(smiles):
        # Check if it's a valid molecule
        if not isinstance(sm, float):
            molecule = Chem.MolFromSmiles(sm)
            entry = catalog.GetMatches(molecule) #Get all the matches/PAINs
            if entry:
                groups = []
                pains = []
                for i in entry:
                    pains.append(i.GetDescription().capitalize())
                    groups.append(i.GetProp('Scope'))
                n_pains.append(len(pains))
            else:
                n_pains.append(0)
        else:
            n_pains.append(4)

    return n_pains

# Calculate the number of PAINS substructures in a dataframe of compounds
def PAINS(df_path, save_path):
    df = pd.read_csv(df_path)
    smiles = list(df['canonical_smiles'])

    n_pains = n_PAINS(smiles)

    df['PAINS'] = n_pains
    df.to_csv(save_path, index=False)

# Initialize vocabulary and autoencoder
def initialize_models(main_path, vocab_path, ae_path):
    # Create Vocab
    df = pd.read_csv(vocab_path)
    selfies = list(df['selfies'])
    vocab = Vocabulary(selfies)
    print("Vocab Done!")
    
    # Load AE
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

    auto = AE(main_path, input_shape, latent_dim, lstm_units, output_dim, batch_norm, batch_norm_momentum, noise_std, numb_dec_layer, embedding_dim, vocab.vocab_size, vocab.max_len)
    auto.load_autoencoder_model(ae_path)
    print("AE Done!")

    return vocab, auto

# Check if a compound is a novel generation, or if it already exists in chembl database
def novelty_check(smiles, all_smiles):
    if smiles in all_smiles:
        return False
    else:
        return True

# Calculate the novelty percentage of a set of generated compounds
def percent_novel(df, df_chembl, save_path, checkpoint_path):
    all_smiles = list(df_chembl['canonical_smiles'])

    checkpoints = open(checkpoint_path, 'w')

    total = len(df)
    counter = 0
    novel = []
    for i in range(len(df)):
        if novelty_check(df.at[i, 'canonical_smiles'], all_smiles):
            counter += 1
            novel.append(True)
            checkpoints.write('True\n')
        else:
            novel.append(False)
            checkpoints.write('False\n')
        print(str(i) + " Done")
    print(counter / total)
    checkpoints.write('Percent Novel: ' + str(counter / total))

    # Save
    df['Novel'] = novel
    df.to_csv(save_path, index=False)

# Method to calculate/predict metrics for generated compounds
# Metrics: MW, QED, SAS, pIC50, logBB, logCmax, logThalf
def calculate_metrics(df_path, save_path, metrics=['MW', 'QED', 'SAS', 'PAINS', 'pIC50', 'logBB', 'logCmax', 'logThalf']):
    # Get molecules
    df = pd.read_csv(df_path)
    selfies = list(df['selfies'])

    # QED and SAS and MW
    if 'MW' in metrics or 'QED' in metrics or 'SAS' in metrics:
        mw = calculate_MW(selfies)
        qed = calculate_qed(selfies)
        sas = calculate_sas(selfies)
        df['MW'] = mw
        df['QED'] = qed
        df['SAS'] = sas
        df.to_csv(save_path, index=False)

    # PAINS
    if 'PAINS' in metrics:
        df = pd.read_csv(df_path)
        smiles = list(df['canonical_smiles'])
        n_pains = n_PAINS(smiles)
        df['PAINS'] = n_pains
        df.to_csv(save_path, index=False)
    
    # Load Vocab and AE for Predictors
    main_path = ''
    vocab_path = 'datasets/500k_small_molecule.csv'
    ae_path = 'models/AE_model.h5'
    predictor_path = ''
    vocab, auto = initialize_models(main_path, vocab_path, ae_path)
    
    lv = None
    if 'pIC50' in metrics or 'logBB' in metrics or 'logCmax' in metrics or 'logThalf' in metrics:
        print('Generating latent vectors...')
        df = pd.read_csv(df_path)
        selfies = list(df['selfies'])
        lv = auto.selfies_to_latentvector(vocab, selfies)
        print('Latent vectors done...')

    # pIC50
    if 'pIC50' in metrics:
        df = pd.read_csv(df_path)
        df_train = pd.read_csv('datasets/BACE1.csv')
        hyperparams = [1,256,0.0001]
        predictor_lv.repurpose_for_target(predictor_path, 'pIC50', vocab, auto, df_train, df, hyperparams, save_path, str=False, lv = lv)

    # logBB
    if 'logBB' in metrics:
        df = pd.read_csv(df_path)
        df_train = pd.read_csv('datasets/logBB.csv')
        hyperparams = [1,256,0.001]
        predictor_lv.repurpose_for_target(predictor_path, 'logBB', vocab, auto, df_train, df, hyperparams, save_path, str=False, lv = lv)

    # Cmax
    if 'logCmax' in metrics:
        df = pd.read_csv(df_path)
        df_train = pd.read_csv('datasets/logCmax.csv')
        hyperparams = [1,1024,0.0001]
        predictor_lv.repurpose_for_target(predictor_path, 'logCmax', vocab, auto, df_train, df, hyperparams, save_path, str=False, lv = lv)

    # T1/2
    if 'logThalf' in metrics:
        df = pd.read_csv(df_path)
        df_train = pd.read_csv('datasets/logthalf.csv')
        hyperparams = [1,512,0.0001]
        predictor_lv.repurpose_for_target(predictor_path, 'logThalf', vocab, auto, df_train, df, hyperparams, save_path, str=False, lv = lv)

# Method to calculate fitness score of compounds
def calculate_fitness(df_path):
    df = pd.read_csv(df_path)
    p = list(df['pIC50'])
    b = list(df['logBB'])
    c = list(df['logCmax'])
    t = list(df['logThalf'])
    fitness = [p[i] + b[i] + c[i] + t[i] for i in range(len(p))] # The fitness function simplifies to this equation
    df['Fitness'] = fitness
    df.to_csv(df_path, index=False)

# Method to pass compounds through SAS-QED-PAINS filter
def filter(df):
    passes = []
    for i in range(len(df)):
        if df.at[i, 'SAS'] < 6 and df.at[i, 'QED'] > 0.4 and df.at[i, 'PAINS'] == 0:
            passes.append(True)
        else:
            passes.append(False)
    return passes

# Method to filter compounds in a dataframe
# Saves all compounds that pass the filter
# Saves a subset of compounds with highest fitness scores that pass the filter
def filter_compounds(df_path, n_subset, save_path_best, save_path_subset):
    df = pd.read_csv(df_path)
    filter_passes = filter(df)
    
    indices = []
    for i in range(len(df)):
        if df.at[i, 'Novel'] and filter_passes[i]:
            indices.append(i)
    
    df = df.loc[indices]
    df.sort_values(by='Fitness', ascending=False, ignore_index=True, inplace=True)
    df.to_csv(save_path_best, index=False)
    print('# Best Compounds: ' + str(len(df)))

    df = df.loc[:n_subset]
    df.to_csv(save_path_subset, index=False)