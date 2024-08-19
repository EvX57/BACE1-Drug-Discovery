import pandas as pd
from chembl_webresource_client.new_client import new_client
import numpy as np
from rdkit import Chem
from rdkit.Chem import RDConfig, PandasTools
import statistics
import selfies as sf

import math
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

# Convert from IC50 to pIC50
def pIC50(input):
    pIC50 = []

    for i in input['standard_value_norm']:
        molar = i*(10**-9) # Converts nM to M
        pIC50.append(-np.log10(molar))

    input['pIC50'] = pIC50
    x = input.drop('standard_value_norm', 1)
        
    return x

# Normalization for pIC50 conversion
def norm_value(input):
    norm = []

    for i in input['standard_value']:
        if float(i) > 100000000:
          i = 100000000
        norm.append(float(i))

    input['standard_value_norm'] = norm
    x = input.drop('standard_value', 1)
        
    return x

# Preview chembl search results
def preview(search):
    # Get Info From Database
    target = new_client.target
    target_query = target.search(search)
    targets = pd.DataFrame.from_dict(target_query)
    print(targets.target_chembl_id)

# Get all information of inhibitors for target protein from Chembl
# Store information in .csv files
def preprocess(search, chembl_id, acronym, save_folder):
    # Get Info From Database
    target = new_client.target
    target_query = target.search(search)
    targets = pd.DataFrame.from_dict(target_query)
    index = list(targets.target_chembl_id).index(chembl_id)
    selected_target = targets.target_chembl_id[index]
    activity = new_client.activity
    res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")
    df = pd.DataFrame.from_dict(res)
    df.to_csv(save_folder + acronym + '_bioactivity_data.csv', index=False)

    # Get Important Vals
    df2 = df[df.standard_value.notna()]
    mol_cid = []
    for i in df2.molecule_chembl_id:
        mol_cid.append(i)
    canonical_smiles = []
    for i in df2.canonical_smiles:
        canonical_smiles.append(i)
    standard_value = []
    for i in df2.standard_value:
        standard_value.append(i)
    data_tuples = list(zip(mol_cid, canonical_smiles, standard_value))
    df3 = pd.DataFrame(data_tuples,  columns=['chembl_id', 'canonical_smiles', 'standard_value'])
    df3.to_csv(save_folder + acronym + '_bioactivity_preprocessed_data.csv', index=False)

    # Remove duplicate canonical SMILES
    # Average standard values
    df3 = remove_duplicate_SMILES_w_standard_value(df3)
    df3.to_csv(save_folder + acronym + '_bioactivity_filtered.csv', index=False)

    # pIC50
    df_norm = norm_value(df3)
    df_final = pIC50(df_norm)

    # QED
    smiles = list(df_final['canonical_smiles'])
    mols = [Chem.MolFromSmiles(sm) for sm in smiles]
    qed = [Chem.QED.default(m) for m in mols]
    df_final['QED'] = qed

    # SAS
    sas = [sascorer.calculateScore(m) for m in mols]
    df_final['SAS'] = sas

    df_final.to_csv(save_folder + acronym + '_master_data.csv', index=False)

# Preprocess inhibitor information in .csv files
# Convert SMILES to SELFIES
# Remove SELFIES longer than 100 characters
# Remove inhibitors with "." in SMILES representation
def preprocess_p2(df_path, save_path):
    df = pd.read_csv(df_path)
    SMILES_to_SELFIES(df, save_path)
    df = pd.read_csv(save_path)
    SELFIES_cutoff(df, save_path)
    df = pd.read_csv(save_path)
    remove_period_SMILES(df, save_path)

# Preprocess the dataset for a metric
def preprocess_metric(df_path, save_path):
    # Read raw .csv file
    df = pd.read_csv(df_path)

    # Preserve important columns in raw .csv file
    cols = ['Molecule ChEMBL ID', 'Smiles', 'Standard Value', 'Standard Units']
    df = df[cols]
    df.rename(columns={'Smiles':'canonical_smiles'}, inplace=True)

    # Convert to canonical SMILES and canonical SELFIES
    # Remove SELFIES exceeding length threshold
    # Remove SMILES containing "."
    make_canonical(df, 'canonical_smiles', save_path)
    df = pd.read_csv(save_path)
    SMILES_to_SELFIES(df, save_path)
    df = pd.read_csv(save_path)
    SELFIES_cutoff(df, save_path)
    df = pd.read_csv(save_path)
    remove_period_SMILES(df, save_path)
    df = pd.read_csv(save_path)

    # Check for duplicate entries
    sm = list(df['canonical_smiles'])
    print(len(sm))
    print(len(list(set(sm))))

# Remove duplicate SMILES and average their pIC50's
def remove_duplicate_SMILES_w_standard_value(df):
    all_smiles = list(df['canonical_smiles'])
    all_ids = list(df['chembl_id'])
    all_values = list(df['standard_value'])
    existing_smiles = []
    repeating_smiles = []

    filtered_id = []
    filtered_sv = []
    for i, sm in enumerate(all_smiles):
        if sm in existing_smiles:
            if sm not in repeating_smiles:
                repeating_smiles.append(sm)
        else:
            existing_smiles.append(sm)
            filtered_id.append(all_ids[i])
            filtered_sv.append(all_values[i])

    for sm in repeating_smiles:
        # Get indices of all occurences of current repeating SMILES
        indices = [i for (i, s) in enumerate(all_smiles) if s==sm]

        # Get all standard values for this repeating smiles
        all = [all_values[i] for i in indices]
        print(all)
        all = [float(v) for v in all]
        print("Length: " + str(len(all)) + "\tStdev: " + str(statistics.stdev(all)) + "")
        avg = statistics.mean(all)
        filtered_sv[existing_smiles.index(sm)] = avg

    # Save to df
    new_df = pd.DataFrame()
    new_df['chembl_id'] = filtered_id
    new_df['canonical_smiles'] = existing_smiles
    new_df['standard_value'] = filtered_sv

    return new_df

# Remove duplicate SMILES values
# Average their metric values
def remove_duplicates(df, metric, save_path, save_columns=[]):
    dict = {}
    for i in range(len(df)):
        sm = df.at[i, 'canonical_smiles']
        m = df.at[i, metric]
        if sm in dict.keys():
            dict[sm].append(m)
        else:
            dict[sm] = [m]

    all_sm = []
    all_m = []
    for k in dict.keys():
        all_sm.append(k)
        all_m.append(statistics.mean(dict[k]))

    df_new = pd.DataFrame()
    df_new[metric] = all_m
    df_new['canonical_smiles'] = all_sm

    if len(save_columns) > 0:
        column_vals = {}
        for c in save_columns:
            column_vals[c] = []

        for sm in all_sm:
            for i in range(len(df)):
                if df.at[i, 'canonical_smiles'] == sm:
                    for c in save_columns:
                        column_vals[c].append(df.at[i, c])
                    break

        for c in save_columns:
            df_new[c] = column_vals[c]

    print(len(df_new))
    df_new.to_csv(save_path, index=False)

# Convert sdf to SMILES
def sdf_to_SMILES(sdf_path, save_path):
    df = PandasTools.LoadSDF(sdf_path)
    molecules = list(df['ROMol'])
    all_sm = []
    for i in range(len(molecules)):
        mol = molecules[i]
        sm = Chem.MolToSmiles(mol)
        all_sm.append(sm)
    df['canonical_smiles'] = all_sm
    df.to_csv(save_path, index=False)

# Convert SMILES to SELFIES
# Remove SMILES that don't follow default SELFIES constraints
# Remove SMILES that aren't properly converted from SMILES --> SELFIES --> SMILES
def SMILES_to_SELFIES(df, save_path):
    columns = list(df.columns)
    all_selfies = []
    counter = 0
    for i in range(len(df)):
        failed = False
        try:
            sm = df.at[i, 'canonical_smiles']
            selfies = sf.encoder(sm)
            dec_sm = sf.decoder(selfies)
            dec_sm = Chem.CanonSmiles(dec_sm)
            sm = Chem.CanonSmiles(sm)
            if dec_sm != sm:
                print("Error at index " + str(i) + ": Failed Match")
                failed = True
                counter += 1
            else:
                all_selfies.append(selfies)
        except sf.exceptions.EncoderError:
            print("Error at index " + str(i) + ": Failed Conversion")
            failed = True
            counter += 1
        if failed:
            df.drop(labels=i, inplace=True)
    df.reset_index(inplace=True, drop=True)
    df['selfies'] = all_selfies
    columns.insert(2, 'selfies')
    df = df[columns]
    df.to_csv(save_path, index=False)
    print('Total Fails: ' + str(counter))

# Select SELFIES with fewer number of characters than the vocabulary threshold
def SELFIES_cutoff(df, save_path, length_threshold=100):
    indices = []
    for i in range(len(df)):
        if sf.len_selfies(df.at[i, 'selfies']) < length_threshold:
            indices.append(i)
    
    new_df = df.loc[indices]
    new_df.reset_index(inplace=True, drop=True)
    new_df.to_csv(save_path, index=False)

# Remove inhibitors containing a "." in SMILES notation
def remove_period_SMILES(df, save_path):
    for i in range(len(df)):
        if '.' in df.at[i, 'canonical_smiles']:
            df.drop(labels=i, inplace=True)
        if (i % 1000) == 0:
            print(i)
    df.reset_index(inplace=True, drop=True)
    df.to_csv(save_path, index=False)

# Sample SELFIES from larger dataset
# Used to construct 500k and 100k small molecule datasets
def sample_SELFIES(df, num, length_threshold):
    # Randomly shuffle dataframe
    df = df.sample(frac=1, ignore_index=True)

    indices = []
    index = 0
    success = 0
    while success < num:
        cur_selfies = df.at[index, 'selfies']
        if sf.len_selfies(cur_selfies) < length_threshold:
            indices.append(index)
            success += 1
        index += 1
    
    new_df = df.loc[indices]
    new_df.reset_index(inplace=True, drop=True)
    return new_df

# Double checks that all smiles are canonical
def make_canonical(df, col_name, save_path):
    smiles = list(df[col_name])

    can_smiles = []
    valid_indices = []
    for i, sm in enumerate(smiles):
        try:
            can_smiles.append(Chem.CanonSmiles(sm))
            valid_indices.append(i)
        except TypeError:
            continue
    
    df = df.loc[valid_indices]
    df[col_name] = can_smiles
    df.to_csv(save_path, index=False)

# Convert a metric to log scale
# Used for Cmax and t1/2
def log_metric(path, name, new_name, save_path):
    df = pd.read_csv(path)
    indices = []
    for i in range(len(df)):
        if df.at[i, name] > 0.0:
            df.at[i, name] = math.log10(df.at[i, name])
            indices.append(i)

    df = df.loc[indices]
    df.rename(columns={name:new_name}, inplace=True)
    df.to_csv(save_path, index=False)