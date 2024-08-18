import os
import pandas as pd
import shutil

# Extract the best docked poses of each compound
# folder_compounds: folder containing pdbqt structures of compounds
# receptor: receptor name
# folder_results: parent folder of folder containing docking results
# num_poses: number of top binding poses to extract from each compound
def extract_poses(folder_compounds, receptor, folder_results, num_poses):
    names = os.listdir(folder_compounds)
    names = [n.split('.pdbqt')[0] for n in names if '.pdbqt' in n and receptor not in n]

    # Loop through each compound
    for name in names:
        # Find location of docking results
        folder = folder_results + 'Results/'
        prefix = 'docking_' + name + '.pdb'
        all_files = os.listdir(folder)
        if prefix in all_files:
            all_files = [prefix]
        else:
            continue
        
        # Create folder to store results
        save_folder = folder_results + name + '/'
        os.mkdir(save_folder)

        # Get binding energies
        all_energies = []
        trial_best_energies = []
        all_fnames = []
        all_models = []
        for f_name in all_files:
            energies = []
            f = open(folder + f_name, 'r')
            current_model = 0
            while True:
                line = f.readline()

                # End of file
                if line == '':
                    break
                # New Model
                if 'MODEL' in line:
                    current_model = int(line.split()[1])
                # Energy
                if 'REMARK INTER + INTRA:' in line:
                    vals = line.split()
                    energy = float(vals[vals.index('INTRA:') + 1])
                    energies.append(energy)
                    all_energies.append(energy)
                    all_fnames.append(f_name)
                    all_models.append(current_model)
            f.close()
            trial_best_energies.append(min(energies))

        # Sort binding energies
        sorted_vals = sorted(zip(all_energies, all_fnames, all_models), key=lambda pair: pair[0])

        # Save to file
        energy_output = open(save_folder + 'binding_energies.txt', 'w')
        energy_output.write('Min: ' + str(min(all_energies)) + '\n')

        for v in sorted_vals:
            energy_output.write(v[1] + '  Model ' + str(v[2]) + ': ' + str(v[0]) + ' (kcal/mol)\n')
        energy_output.close()

        # Extract top poses
        for i in range(num_poses):
            _, f_name, model = sorted_vals[i]
            m_name = 'MODEL ' + str(model)
            f = open(folder + f_name, 'r')
            output = open(save_folder + name + '_pose_' + str(i+1) + '.pdb', 'w')

            reading = False
            while True:
                line = f.readline()

                if reading:
                    # Stop reading at end of file or next model
                    if line == '' or 'MODEL' in line:
                        f.close()
                        output.close()
                        break
                    else:
                        output.write(line)
                else:
                    if m_name in line:
                        reading = True
                    if line == '':
                        f.close()
                        output.close()
                        break

# Sort the top docked poses across all compounds by binding energy
# folder_compounds: folder containing pdbqt structures of compounds
# receptor: receptor name
# folders_results: list of parent folders of folders containing docking results
# save_path: save location of sorted results
def sort_compounds(folder_compounds, receptor, folders_results, save_path):
    prefixes = os.listdir(folder_compounds)
    prefixes = [n.split('.pdbqt')[0] for n in prefixes if '.pdbqt' in n and receptor not in n]

    df = pd.DataFrame(columns=['Name', 'Min'])
    for folder in folders_results:
        all_files = os.listdir(folder)
        for f in all_files:
            for prefix in prefixes:
                if prefix == f:
                    file = open(folder + f + '/binding_energies.txt')

                    vals = file.readline()
                    vals = vals.split()
                    min = vals[1]

                    df.loc[len(df)] = [f, float(min)]

    df.sort_values(by='Min', ascending=True, inplace=True)
    df.to_csv(save_path, index=False)

# Helper method for compare_to_CTCD_BE
# Counts the number of values that surpass a threshold
def threshold_counter_BE(threshold, values):
    values.append(threshold)
    values.sort()
    return values.index(threshold)

# Compare binding energy results between generated compounds and clinical trial candidate drugs
# gen_path: .csv file of binding energy results for generated compounds
# CTCD_path: .csv file of binding energy results for clinical trial candidate drugs
def compare_to_CTCD_BE(gen_path, CTCD_path):
    df_CTCD = pd.read_csv(CTCD_path)
    
    threshold_all = df_CTCD.at[0, 'Min']
    threshold_min = df_CTCD.at[len(df_CTCD)-1, 'Min']

    df_gen = pd.read_csv(gen_path)
    vals = list(df_gen['Min'])

    n_all = threshold_counter_BE(threshold_all, vals)
    n_min = threshold_counter_BE(threshold_min, vals)

    print('Better Than Worst: ' + str(n_min))
    print('Better Than All: ' + str(n_all))

    return n_all

# Extract the generated compounds that outperform CTCD binding energy results
# gen_path: .csv file of binding energy results for generated compounds
# CTCD_path: .csv file of binding energy results for clinical trial candidate drugs
# folders_compounds: list of folders containing pdbqt structures of compounds
# save_folder: save location for pdbqt structures of extracted compounds
def extract_best_compounds_BE(gen_path, CTCD_path, folders_compounds, save_folder):
    n = compare_to_CTCD_BE(gen_path, CTCD_path)

    df = pd.read_csv(gen_path)
    names = list(df['Name'])[:n]

    for n in names:
        for f in folders_compounds:
            files = os.listdir(f)
            prefix = n + '.pdbqt'
            if prefix in files:
                shutil.copyfile(f + prefix, save_folder + prefix)

# Calculate selectivity score of compounds
# p_bace1: .csv file of bace1 binding energy results
# p_bace2: .csv file of bace2 binding energy results
# save_path: save location for selectivity score results
def calculate_selectivity_score(p_bace1, p_bace2, save_path):
    df_bace1 = pd.read_csv(p_bace1)
    df_bace2 = pd.read_csv(p_bace2)

    df_bace1.set_index('Name', inplace=True)

    ratios = []
    for i in range(len(df_bace2)):
        name = df_bace2.at[i, 'Name']
        be_2 = df_bace2.at[i, 'Min']
        be_1 = df_bace1.at[name, 'Min']
        ratio = be_1 / be_2
        ratios.append(ratio)
    
    df_bace2['BE Ratio'] = ratios
    df_bace2.sort_values(by='BE Ratio', ascending=False, inplace=True)
    df_bace2.to_csv(save_path, index=False)

# Helper method for compare_to_CTCD_SS
# Counts the number of values that surpass a threshold
def threshold_counter_SS(threshold, values):
    values.append(threshold)
    values.sort(reverse=True)
    return values.index(threshold)

# Compare selectivity score results between generated compounds and clinical trial candidate drugs
# gen_path: .csv file of selectivity score results of generated compounds
# CTCD_path: .csv file of selectivity score results of clinical trial candidate drugs
def compare_to_CTC_SS(gen_path, CTCD_path):
    df_CTCD = pd.read_csv(CTCD_path)
    
    threshold_all = df_CTCD.at[0, 'BE Ratio']
    threshold_min = df_CTCD.at[len(df_CTCD)-1, 'BE Ratio']

    df = pd.read_csv(gen_path)
    vals = list(df['BE Ratio'])

    n_all = threshold_counter_SS(threshold_all, vals)
    n_min = threshold_counter_SS(threshold_min, vals)

    print('Better Than Worst: ' + str(n_min))
    print('Better Than All: ' + str(n_all))

    return n_all

# Select the generated compounds that outperform CTCD selectivity score results
# These are the candidate inhibitors
# gen_path: .csv file of selectivity score results of generated compounds
# CTCD_path: .csv file of selectivity score results of clinical trial candidate drugs
# p_bace1: .csv file of bace1 binding energy results of generated compounds
# save_path: save location of results
def select_best_compounds_SS(gen_path, CTCD_path, p_bace1, save_path):
    n = compare_to_CTC_SS(gen_path, CTCD_path)

    df = pd.read_csv(gen_path)
    df = df.truncate(after=n-1)

    df_be = pd.read_csv(p_bace1)
    df_be.set_index(keys='Name', inplace=True)
    ref_vals = []
    for i in range(len(df)):
        name = df.at[i, 'Name']
        ref_vals.append(df_be.at[name, 'Min'])
    
    df['BACE1'] = ref_vals
    df.rename(columns={'Min':'BACE2'}, inplace=True)
    df = df[['Name', 'BACE1', 'BACE2', 'BE Ratio']]

    df.to_csv(save_path, index=False)

# Extract the best docked poses of candidate inhibitors
# df_path: .csv file of candidate inhibitors
# folders_results: list of parent folders of folders containing docking results
# save_folder: save location of extracted poses
def extract_candidate_inhibitor_poses(df_path, folders_results, save_folder):
    df = pd.read_csv(df_path)
    names = list(df['Name'])

    for n in names:
        for folder in folders_results:
            if n in os.listdir(folder):
                src = folder + n + '/' + n + '_pose_1.pdb'
                dest = save_folder + n + '.pdb'
                shutil.copyfile(src, dest)