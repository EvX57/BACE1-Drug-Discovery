import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
from rdkit.Chem import Draw, MolFromSmiles

# Compare distributions of a single property
def compare_property_distribution(vals, names, x_label, title, save_path):
    #colors = ['#1f77b4', '#ff7f0e', '#d62728', '#bcbd22', '#17becf']
    #colors = ['#cfcfcf', '#a3b9c8', '#77a3c1', '#4a8dbb', '#1f77b4']
    colors = ['#7f7f7f', '#ff7f0e', '#1f77b4']
    #colors = ['#000000', '#7f7f7f', '#ff7f0e']
    #colors = ['#000000', '#ff7f0e']
    fw = 'regular'

    for i in range(len(vals)):
        sns.kdeplot(data=vals[i], fill=True, label=names[i], color=colors[i])
    
    # if x_label == 'logCmax':
    #     text = 'log(' + r'$C_{max}$' + ')'
    # elif x_label == 'logThalf':
    #     text = 'log(' + r'$t_{1/2}$' + ')'
    # else:
    #     text = x_label
    # plt.suptitle(text + ' Distribution', fontweight=fw, size=18)
    # plt.xlabel(text, fontweight=fw, size=14)

    plt.suptitle(title, fontweight=fw, size=18)
    plt.xlabel(x_label, fontweight=fw, size=14)
    plt.ylabel('Density', fontweight=fw, size=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Compare distributions of multiple properties
def compare_all_property_distributions(dfs, names, save_folder):
    #properties = ['pIC50', 'logBB', 'logCmax', 'logThalf']
    properties = ['pIC50', 'MW']

    for p in properties:
        vals = []
        for i in range(len(dfs)):
            v = list(dfs[i][p])

            # Remove outliers from predicted values for visualization purposes
            if p == 'pIC50':
                v = [x for x in v if x > 0]
            elif p == 'logBB':
                v = [x for x in v if x < 5 and x > -5]
            elif p == 'logCmax':
                v = [x for x in v if x > -3]
            elif p == 'logThalf':
                v = [x for x in v if x < 6 and x > -4]
            vals.append(v)
        compare_property_distribution(vals, names, p, p + ' Distribution', save_folder + p + '_distribution.png')

# Save the best generated and novel molecules to an image
def print_best_molecules(n_mols, df_mols, save_folder):
    i = 0
    counter = 0
    while i < len(df_mols) and counter < n_mols:
        if df_mols.at[i, 'Novel']:
            sm = df_mols.at[i, 'canonical_smiles']
            mol = MolFromSmiles(sm)
            Draw.MolToFile(mol, save_folder + 'molecule_' + str(i+1) + '.png')
            counter += 1
        i += 1

# Plot a scatter plot of SAS and QED values
# Contains density plots on the axes
def qed_sas_scatter(dfs, number, save_path):
    colors = ['#ff7f0e', '#1f77b4']

    qed_vals = []
    sas_vals = []
    insides = []
    outsides = []
    for df in dfs:
        df = df.sample(frac=1)
        qed = list(df['QED'])
        sas = list(df['SAS'])
        qed = qed[:number]
        sas = sas[:number]

        # Sort to inside/outside boundary
        # Boundary: 0.5<=QED<=1.0 & 0<=SAS<=5
        inside = [[] for _ in range(2)]
        outside = [[] for _ in range(2)]
        for i in range(len(qed)):
            if qed[i] > 0.4 and qed[i] <= 1 and sas[i] >= 0 and sas[i] < 6:
                inside[0].append(qed[i])
                inside[1].append(sas[i])
            else:
                outside[0].append(qed[i])
                outside[1].append(sas[i])
        qed_vals.append(qed)
        sas_vals.append(sas)
        insides.append(inside)
        outsides.append(outside)

    fig, axs=plt.subplots(2,2,figsize=(8,6),gridspec_kw={'hspace': 0, 'wspace': 0,'width_ratios': [7, 1], 'height_ratios': [1, 7]})
    axs[0,0].axis("off")
    axs[0,1].axis("off")
    axs[1,1].axis("off")

    for i in range(len(dfs)):
        sns.scatterplot(x=outsides[i][1], y=outsides[i][0], s=1.5, color=colors[i], ax=axs[1,0], marker='X')
        sns.scatterplot(x=insides[i][1], y=insides[i][0], s=1.5, color=colors[i], ax=axs[1,0])

    axs[1,0].add_patch(patches.Rectangle(xy=[0,0.4], width=6, height=0.6, linewidth=0.75, color='black', fill=False))

    axs[1,0].set_xlabel('SAS', fontsize=11, labelpad=7.0)
    axs[1,0].set_ylabel('QED', fontsize=11, labelpad=7.0)
    axs[1,0].set_xticks([0,2,4,6,8])
    axs[1,0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])

    for i in range(len(dfs)):
        sns.distplot(sas_vals[i], hist=True, kde=False, color=colors[i], ax=axs[0,0])
        sns.distplot(qed_vals[i], hist=True, kde=False, color=colors[i], ax=axs[1,1], vertical=True)
    fig.suptitle('Generated Compounds', fontsize=15, y=0.94)
    fig.legend(labels=['TL', 'GA'], markerscale=5.0, loc='upper right', bbox_to_anchor=(0.95, 0.95))
    plt.savefig(save_path)
    plt.close()

# Plot a distribution as a histogram or density plot
def plot_distribution(vals, title, xlabel, save_path, hist=False):
    if hist:
        sns.histplot(data=vals, bins=50)
    else:
        sns.kdeplot(data=vals, fill=True)
    plt.suptitle(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel('Count', fontsize=11)
    plt.savefig(save_path)
    plt.tight_layout()
    plt.close()

# Plot a bar graph
def plot_bar_2(val_sets, names, title, xlabel, ylabel, save_path):
    colors = ['#ff7f0e', '#1f77b4']

    # Get counts
    dicts = []
    for vals in val_sets:
        dict = {}
        for v in vals:
            if v in dict.keys():
                dict[v] += 1
            else:
                dict[v] = 1
        dicts.append(dict)

    # Save to df
    df = pd.DataFrame(columns=['Model', xlabel, ylabel])
    for i in range(len(dicts)):
        x = list(dicts[i].keys())
        y = list(dicts[i].values())
        for j in range(len(x)):
            df.loc[len(df) - 1] = [names[i], x[j], y[j]]
    
    # Plot
    sns.set_palette(colors)
    g = sns.barplot(data=df, x=xlabel, y=ylabel, hue='Model')
    g.legend().set_title("")
    plt.suptitle(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path)