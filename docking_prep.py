from openbabel import pybel as pb
from openbabel import openbabel as ob
import MDAnalysis as mda
import pandas as pd
import os

# Convert compounds from SMILES strings to pdbqt structures
def smiles_to_pdbqt(smi, names, save_folder, hyperparams=[100, 100, 250]):
    # Convert
    for i, sm in enumerate(smi):
        mol = pb.readstring('smiles', sm)

        # generate 3D coordinates - handles adding H atoms
        mol.make3D()

        # conformer searching
        ff = pb._forcefields["mmff94"]
        success = ff.Setup(mol.OBMol)
        if not success:
            ff = pb._forcefields["uff"]
            success = ff.Setup(mol.OBMol)
            if not success:
                print("ERROR")

        # forcefield optimization
        print('Initial Energy: ' + str(ff.Energy()))
        ff.ConjugateGradients(hyperparams[0], 1.0e-3)
        ff.FastRotorSearch(True) # permute central bonds
        ff.WeightedRotorSearch(hyperparams[1], 25) # 1000 cycles, each with 25 forcefield ops
        # final optimization
        ff.ConjugateGradients(hyperparams[2], 1.0e-4)
        # update the coordinates
        ff.GetCoordinates(mol.OBMol)
        print('Final Energy: ' + str(ff.Energy()) + '\n')

        # add partial charges
        ob_charge_model = ob.OBChargeModel.FindType("eem2015bn")
        ob_charge_model.ComputeCharges(mol.OBMol)

        # write molecule to file
        mol.write('pdbqt', save_folder + names[i] + '.pdbqt', overwrite=True)

# Define search space box around entire protein
# Used for initial docking of clinical trial candidate drugs to identify active site
def full_protein_box(protein_pdb_file, save_prefix):
    u = mda.Universe(protein_pdb_file)
    g_center = u.atoms.center_of_geometry()

    # Find range of box
    positions = u.atoms.positions

    # Reshape
    reshaped = []
    for j in range(len(positions[0])):
        cur = []
        for i in range(len(positions)):
            cur.append(positions[i][j])
        reshaped.append(cur)

    for i in range(len(reshaped)):
        print(abs(min(reshaped[i]) - g_center[i]))
        print(abs(max(reshaped[i]) - g_center[i]))

    center = []
    lengths = []
    for i in range(len(reshaped)):
        mi = min(reshaped[i])
        ma = max(reshaped[i])
        center.append((mi + ma) / 2)
        lengths.append(ma - mi)

    save_path = save_prefix + '.txt'
    output = open(save_path, 'w')
    output.write('center_x = ' + str(center[0]) + '\n')
    output.write('center_y = ' + str(center[1]) + '\n')
    output.write('center_z = ' + str(center[2]) + '\n')
    output.write('size_x = ' + str(lengths[0]) + '\n')
    output.write('size_y = ' + str(lengths[1]) + '\n')
    output.write('size_z = ' + str(lengths[2]) + '\n')
    output.write('\nnum_modes = 5')
    output.close()

    create_box_pdb(center, lengths, save_prefix + '.pdb')

# Determine the protein active site based on docked poses of clinical trial candidate drugs to entire protein
# folder: contains docking results of all clinical trial candidate drugs
def find_active_site(folder, save_prefix):
    df = pd.read_csv('datasets/BACE1_clinical_trial_candidate_drugs.csv')
    names = list(df['Name'])
    names = [n[:5] for n in names]
    
    # Get all coordinates
    all_atomic_coordinates = [[], [], []]
    for n in names:
        f = open(folder + n + '/' + n + '_pose_1.pdb', 'r')
        
        line = f.readline()
        while line != '':
            if 'ATOM' in line:
                vals = line.split()
                
                x = float(vals[5])
                y = float(vals[6])
                z = float(vals[7])

                all_atomic_coordinates[0].append(x)
                all_atomic_coordinates[1].append(y)
                all_atomic_coordinates[2].append(z)
            line = f.readline()
        
        f.close()

    # Define box from coordinates
    x_min = min(all_atomic_coordinates[0])
    x_max = max(all_atomic_coordinates[0])
    y_min = min(all_atomic_coordinates[1])
    y_max = max(all_atomic_coordinates[1])
    z_min = min(all_atomic_coordinates[2])
    z_max = max(all_atomic_coordinates[2])

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2

    x_len = (x_max - x_min)
    y_len = (y_max - y_min)
    z_len = (z_max - z_min)

    # 2ZHT
    # x_len = 15.0
    # y_len = 20.0
    # z_len = 15.0

    # 6UJ0
    # x_len = 15.0
    # y_len = 15.0
    # z_len = 25.0

    # Write box file
    output = open(save_prefix + '.txt', 'w')
    output.write('center_x = ' + str(x_center) + '\n')
    output.write('center_y = ' + str(y_center) + '\n')
    output.write('center_z = ' + str(z_center) + '\n')
    output.write('size_x = ' + str(x_len) + '\n')
    output.write('size_y = ' + str(y_len) + '\n')
    output.write('size_z = ' + str(z_len) + '\n')
    output.write('\nnum_modes = 5')
    output.close()

    # Create box pdb
    create_box_pdb([x_center, y_center, z_center], [x_len, y_len, z_len], save_prefix + '.pdb')

# Create pdb structure of a box for visualization
def create_box_pdb(centers, lengths, save_path):
    f = open(save_path, 'w')

    f.write('HEADER    CORNERS OF BOX\n')
    f.write('REMARK    CENTER (X Y Z)   ' + str(round(centers[0], 3)) + '  ' + str(round(centers[1], 3)) + '  ' + str(round(centers[2], 3)) + '\n')
    f.write('REMARK    DIMENSIONS (X Y Z)   ' + str(round(lengths[0], 3)) + '  ' + str(round(lengths[1], 3)) + '  ' + str(round(lengths[2], 3)) + '\n')
    
    coordinates = [[], [], []]
    for i in range(3):
        coordinates[i].append(centers[i] - lengths[i] / 2)
        coordinates[i].append(centers[i] + lengths[i] / 2)

    corners = []
    corners.append([coordinates[0][0], coordinates[1][0], coordinates[2][0]])
    corners.append([coordinates[0][1], coordinates[1][0], coordinates[2][0]])
    corners.append([coordinates[0][1], coordinates[1][0], coordinates[2][1]])
    corners.append([coordinates[0][0], coordinates[1][0], coordinates[2][1]])
    corners.append([coordinates[0][0], coordinates[1][1], coordinates[2][0]])
    corners.append([coordinates[0][1], coordinates[1][1], coordinates[2][0]])
    corners.append([coordinates[0][1], coordinates[1][1], coordinates[2][1]])
    corners.append([coordinates[0][0], coordinates[1][1], coordinates[2][1]])

    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for i in range(len(corners)):
        f.write('ATOM      ' + str(i+1) + '  DU' + letters[i] + ' BOX     1      ' + str(round(corners[i][0], 3)) + '  ' + str(round(corners[i][1], 3)) + '  ' + str(round(corners[i][2], 3)) + '\n')

    f.write('CONECT    1    2    4    5\n')
    f.write('CONECT    2    1    3    6\n')
    f.write('CONECT    3    2    4    7\n')
    f.write('CONECT    4    1    3    8\n')
    f.write('CONECT    5    1    6    8\n')
    f.write('CONECT    6    2    5    7\n')
    f.write('CONECT    7    3    6    8\n')
    f.write('CONECT    8    4    5    7\n')

    f.close()

# Create the bash script to run AutoDock Vina simulations
def create_ADV_script(compounds_folder, receptor):
    all_names = os.listdir(compounds_folder)
    all_names = [n.split('.pdbqt')[0] for n in all_names if '.pdbqt' in n and receptor not in n]

    script_vina = open(compounds_folder + 'script_vina.sh', 'w')
    script_vina.write('#!/bin/bash\n')
    for name in all_names:
        script_vina.write('vina --receptor ' + receptor + '.pdbqt --ligand ' + name + '.pdbqt --config box_' + receptor + '.txt --exhaustiveness=128 --spacing 0.1 --out Results/docking_' + name + '.pdb\n')
    script_vina.close()