import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.calibration import calibration_curve
import matplotlib.colors as mcolors
import matplotlib as mpl

class UCBerkeley:
    """From https://brand.berkeley.edu/colors/"""

    info = [
        {"hex_value": "#002676", "name": "Berkeley blue", "type": "primary"},
        {"hex_value": "#018943", "name": "Green Medium", "type": "primary"},
        {"hex_value": "#FDB515", "name": "California Gold", "type": "primary"},
        {"hex_value": "#E7115E", "name": "Rose Medium", "type": "primary"},
        {"hex_value": "#6C3302", "name": "South Hall", "type": "primary"},
        {"hex_value": "#FF0000", "name": "Red", "type": "primary"},
    ]
    colors = [d["hex_value"] for d in info]

style_params = {
    "axes.grid": True,
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.bottom": False,
    "axes.facecolor": "#ebebeb",
    "axes.axisbelow": True,
    "axes.titlelocation": "center",
    "grid.color": "white",
    "grid.linestyle": "-",
    "grid.linewidth": 1,
    "grid.alpha": 1,
    "xtick.color": "#4D4D4D",
    "ytick.color": "#4D4D4D",
    "text.color": "#000000",
    "font.family": ["arial"],
    "image.cmap": "viridis",
    "axes.prop_cycle": mpl.cycler(color=UCBerkeley.colors),
}
mpl.rcParams.update(style_params)

def generate_table_zeroshot_mimic(models,labels,group):
    with open("./reports/zeroshot_mimic.txt", "w") as text_file:
        text_file.write("\\begin{table}[H]\n")
        text_file.write("\t\\centering\n")

        column_format = "c"+"|cc"*(len(labels)+1) #+1 because of the mean column
        text_file.write("\t\\begin{tabular}{"+column_format+"|}\n")

        column_header = "\t"
        for label in labels:
            column_header += "& \multicolumn{2}{c}{"+label+"} "
        column_header += "& \multicolumn{2}{c}{Mean} "
        text_file.write(column_header+r"\\"+"\n")

        text_file.write("\t"+"& AUC & AUPRC "*(len(labels)+1)+r"\\ \hline"+"\n")


        for model_name in models:
            df = pd.read_csv(f"./data/performance/MIMIC/zeroshot_{model_name}.csv")
            line_metric = f"\t{models[model_name]} "
            line_ci="\t"
            for l in labels:
                auc = df[df["class"].isin([l]) & df["group"].isin([group])]["AUC"].item()
                auc_ci_low = df[df["class"].isin([l]) & df["group"].isin([group])]["CI_AUC_low"].item()
                auc_ci_up = df[df["class"].isin([l]) & df["group"].isin([group])]["CI_AUC_up"].item()

                auprc = df[df["class"].isin([l]) & df["group"].isin([group])]["AUPRC"].item()
                auprc_ci_low = df[df["class"].isin([l]) & df["group"].isin([group])]["CI_AUPRC_low"].item()
                auprc_ci_up = df[df["class"].isin([l]) & df["group"].isin([group])]["CI_AUPRC_up"].item()

                line_metric += f'& {auc} & {auprc} '
                line_ci += f'& [{auc_ci_low},{auc_ci_up}] & [{auprc_ci_low},{auprc_ci_up}]'
            line_metric += f'& {df[df["group"].isin([group])]["AUC"].mean().round(2)} & {df[df["group"].isin([group])]["AUPRC"].mean().round(2)} '
            line_metric += r"\\"
            line_metric += "\n"
            text_file.write(line_metric)

            line_ci += f'& \pm {df[df["group"].isin([group])]["AUC"].std().round(2)} & \pm {df[df["group"].isin([group])]["AUPRC"].std().round(2)} '
            line_ci += r"\\"
            line_ci += "\n"
            text_file.write(line_ci)
        text_file.write("\t\\hline\n")
        text_file.write("\t\\end{tabular}\n")
        text_file.write("\t\\caption{AUC and adjusted AUPRC of zeroshot classification.}\n")
        text_file.write("\t\\label{tab:zeroshot_perf}\n")
        text_file.write("\\end{table}\n")

def generate_barplot_subgroup(models,dataset,groups,subgroup_name,x_label=True):
    print(subgroup_name)
    for model_name in models:
        print(model_name)
        Path(f"./reports/figures/subgroups_perf/{model_name}").mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(f"./data/performance/{dataset}/zeroshot_{model_name}.csv")
        df_res = df[df["group"].isin(groups)]
        df_res['group'] = pd.Categorical(df_res['group'], groups)
        df_res["CI_AUC_low"] = df_res["AUC"] - df_res["CI_AUC_low"]
        df_res["CI_AUC_up"] = df_res["CI_AUC_up"] - df_res["AUC"]
        df_res["CI_AUPRC_low"] = abs(df_res["AUPRC"] - df_res["CI_AUPRC_low"])
        df_res["CI_AUPRC_up"] = abs(df_res["CI_AUPRC_up"] - df_res["AUPRC"])
        df_res = df_res.sort_values(['group','class'])
        plt.figure(figsize=(10,5))
        ax = sns.barplot(data=df_res, x='class', y='AUC', hue='group',hue_order=groups)
        ax.legend_.remove()
        ax.tick_params(axis='y',labelsize=20)
        ax.set_ylabel("AUC",fontdict = {'fontsize' : 20})
        plt.title(f"AUC per subgroup of {subgroup_name}",fontdict = {'fontsize' : 30})
        ax.set_ylim(0,1)
        
        if x_label:
            ax.set_xlabel("Class",fontdict = {'fontsize' : 20})
            ax.tick_params(axis='x',labelsize=20)
            plt.xticks(rotation=25)

        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")

        x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches][:len(df_res)]
        y_coords = [p.get_height() for p in ax.patches][:len(df_res)]
        yerr = list(np.array([df_res["CI_AUC_low"].tolist(),df_res["CI_AUC_up"].tolist()]))

        ax.errorbar(x=x_coords, y=y_coords, yerr=yerr, fmt="none", c="k")
        plt.savefig(f"./reports/figures/subgroups_perf/{model_name}/{subgroup_name}_{model_name}_auc.png", bbox_inches='tight', dpi=300)
        plt.close()

        plt.figure(figsize=(10,5))
        ax = sns.barplot(data=df_res, x='class', y='AUPRC', hue='group',hue_order=groups)
        ax.tick_params(axis='y',labelsize=20)
        ax.set_ylabel(r"$AUPRC_{adj}$",fontdict = {'fontsize' : 20})
        ax.set_ylim(0,1)

        plt.title(r"$AUPRC_{adj}$"+f" per subgroup of {subgroup_name}",fontdict = {'fontsize' : 30})
        
        if x_label:
            ax.set_xlabel("Class",fontdict = {'fontsize' : 20})
            ax.tick_params(axis='x',labelsize=20)
            plt.xticks(rotation=25)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        
        x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches][:len(df_res)]
        y_coords = [p.get_height() for p in ax.patches][:len(df_res)]
        yerr = list(np.array([df_res["CI_AUPRC_low"].tolist(),df_res["CI_AUPRC_up"].tolist()]))
        ax.errorbar(x=x_coords, y=y_coords, yerr=yerr, fmt="none", c="k")
        plt.legend(fontsize=25,loc='center left',bbox_to_anchor=(1, 0.5))
        plt.savefig(f"./reports/figures/subgroups_perf/{model_name}/{subgroup_name}_{model_name}_auprc.png", bbox_inches='tight', dpi=300)
        plt.close()


def generate_barplot_drains(models):
    lst_df = []
    for model_name in models:
        df_model = pd.read_csv(f"./data/performance/CXR14/zeroshot_{model_name}.csv")
        df_model["group"] = df_model["group"].astype(str)
        df_model["group"] = df_model["group"].replace("0","No drain")
        df_model["group"]= df_model["group"].replace("0.0","No drain")
        df_model["group"]= df_model["group"].replace("1","With drain")
        df_model["group"]= df_model["group"].replace("1.0","With drain")
        df_model["model"] = models[model_name]
        lst_df.append(df_model)
    
    df_results_combined = pd.concat(lst_df, axis=0, ignore_index=True)
    plt.figure(figsize=(10,5))
    ax = sns.barplot(data=df_results_combined, x='model', y='AUC', hue='group')
    ax.set_ylabel(r"AUC",fontdict = {'fontsize' : 25})
    ax.set_xlabel("")
    # plt.title(f"AUC of models on images with and without chest drains")
    ax.set_ylim(0,1)
    ax.tick_params(labelsize=25)
    plt.xticks(rotation=45)

    ax.legend_.remove()
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches][:len(df_results_combined)]
    y_coords = [p.get_height() for p in ax.patches][:len(df_results_combined)]
    df_results_combined["CI_AUC_low"] = df_results_combined["AUC"] - df_results_combined["CI_AUC_low"]
    df_results_combined["CI_AUC_up"] = df_results_combined["CI_AUC_up"] - df_results_combined["AUC"]
    yerr = list(np.array([df_results_combined["CI_AUC_low"].tolist(),df_results_combined["CI_AUC_up"].tolist()]))
    ax.errorbar(x=x_coords, y=y_coords, yerr=yerr, fmt="none", c="k")
    plt.savefig(f"./reports/figures/subgroups_perf/pneumothorax_drains_auc.png", bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(10,5))
    ax = sns.barplot(data=df_results_combined, x='model', y='AUPRC', hue='group')
    ax.set_ylabel(r"$AUPRC_{adj}$",fontdict = {'fontsize' : 25})
    ax.set_xlabel("")
    # plt.title(r"$\widehat{AUPRC}$ of models on images with and without chest drains")
    ax.set_ylim(0,1)
    ax.tick_params(labelsize=25)
    plt.xticks(rotation=45)

    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches][:len(df_results_combined)]
    y_coords = [p.get_height() for p in ax.patches][:len(df_results_combined)]
    df_results_combined["CI_AUPRC_low"] = df_results_combined["AUPRC"] - df_results_combined["CI_AUPRC_low"]
    df_results_combined["CI_AUPRC_up"] = df_results_combined["CI_AUPRC_up"] - df_results_combined["AUPRC"]
    yerr = list(np.array([df_results_combined["CI_AUPRC_low"].tolist(),df_results_combined["CI_AUPRC_up"].tolist()]))
    ax.errorbar(x=x_coords, y=y_coords, yerr=yerr, fmt="none", c="k")
    plt.legend(fontsize=25,)
    plt.savefig(f"./reports/figures/subgroups_perf/pneumothorax_drains_auprc.png", bbox_inches='tight', dpi=300)
    plt.close()

def generate_calibration_curves(models, subgroup=True):
    colors = list(mcolors.TABLEAU_COLORS)
    plt.figure()
    plt.plot([0, 1], 
         [0, 1], 
         linestyle='dotted', 
         label='Perfectly Calibrated')
    for i,model_name in enumerate(models):
        df_probas = pd.read_csv(f"./data/probas_CXR14/probas_CXR14_{model_name}.csv")
        y_true = df_probas["Pneumothorax"]
        y_proba = df_probas["proba_Pneumothorax"]
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        
        if subgroup:
            plt.plot(prob_pred,prob_true,linestyle='solid',marker='o',linewidth=1,color=colors[i],label=f'{models[model_name]}, all')
            
            y_true = df_probas[df_probas["Drain"]==1]["Pneumothorax"]
            y_proba = df_probas[df_probas["Drain"]==1]["proba_Pneumothorax"]
            prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
            plt.plot(prob_pred,prob_true,linestyle='dashed',marker='^',linewidth=1,color=colors[i],label=f'{models[model_name]}, only drain')

            y_true = df_probas[df_probas["Drain"]==0]["Pneumothorax"]
            y_proba = df_probas[df_probas["Drain"]==0]["proba_Pneumothorax"]
            prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
            plt.plot(prob_pred,prob_true,linestyle='dashdot',marker='s',linewidth=1,color=colors[i],label=f'{models[model_name]}, no drain')
        else:
            plt.plot(prob_pred,prob_true,linestyle='solid',marker='o',linewidth=1,color=colors[i],label=f'{models[model_name]}')

    # plt.title('Calibration curves for all CLIP-based models')
    plt.xlabel('Mean predicted probability',fontdict = {'fontsize' : 20})
    plt.ylabel('Fraction of positives',fontdict = {'fontsize' : 20})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(f"./reports/figures/calibration_cxr14.png", bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    #Define the models and labels used in the tables/figures
    models = {
        "medclip":"MedCLIP",
        "biovil":"Biovil",
        "biovil-t":"Biovil-t",
        "medimageinsight":"MedImageInsight",
        "chexzero":"CheXzero",
        "cxrclip":"CXR-CLIP"
    }

    labels = [
        'Atelectasis',
        'Cardiomegaly',
        'Consolidation',
        'Pleural Effusion',
        'Pneumonia',
        'Pneumothorax',
    ]
    generate_table_zeroshot_mimic(models,labels,"global")
    generate_barplot_subgroup(models,"MIMIC",["global","Female","Male"],"sex",x_label=False)
    generate_barplot_subgroup(models,"MIMIC",["global","White","Black","Asian"],"race",x_label=True)
    generate_barplot_subgroup(models,"MIMIC",["global","18-25","25-50","50-65","65-80","80+"],"age",x_label=False)
    generate_barplot_drains(models)
    generate_calibration_curves(models,subgroup=False)