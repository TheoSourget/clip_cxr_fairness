import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.calibration import calibration_curve

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
            line = f"\t{models[model_name]} "
            for l in labels:
                line += f'& {df[df["class"].isin([l]) & df["group"].isin([group])]["AUC"].item()} & {df[df["class"].isin([l]) & df["group"].isin([group])]["AUPRC"].item()} '
            line += f'& {df[df["group"].isin([group])]["AUC"].mean().round(2)} & {df[df["group"].isin([group])]["AUPRC"].mean().round(2)} '
            line += r"\\"
            line += "\n"
            text_file.write(line)
        text_file.write("\t\\hline\n")
        text_file.write("\t\\end{tabular}\n")
        text_file.write("\t\\caption{AUC and AUCPR of zeroshot classification}\n")
        text_file.write("\t\\label{tab:zeroshot_perf}\n")
        text_file.write("\\end{table}\n")

def generate_barplot_subgroup(models,dataset,groups,subgroup_name):
    print(subgroup_name)
    for model_name in models:
        Path(f"./reports/figures/subgroups_perf/{model_name}").mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(f"./data/performance/{dataset}/zeroshot_{model_name}.csv")
        df_res = df[df["group"].isin(groups)]
        plt.figure(figsize=(10,10))
        ax = sns.barplot(data=df_res, x='class', y='AUC', hue='group')
        plt.title(f"AUC of {model_name} on the different subgroups of {subgroup_name}")
        ax.set_ylim(0,1)

        x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches][:len(df_res)]
        y_coords = [p.get_height() for p in ax.patches][:len(df_res)]
        df_res["CI_AUC_low"] = df_res["AUC"] - df_res["CI_AUC_low"]
        df_res["CI_AUC_up"] = df_res["CI_AUC_up"] - df_res["AUC"]
        yerr = list(np.array([df_res["CI_AUC_low"].tolist(),df_res["CI_AUC_up"].tolist()]))
        ax.errorbar(x=x_coords, y=y_coords, yerr=yerr, fmt="none", c="k")
        plt.savefig(f"./reports/figures/subgroups_perf/{model_name}/{subgroup_name}_{model_name}_auc.png", bbox_inches='tight', dpi=300)
        plt.close()

        plt.figure(figsize=(10,10))
        ax = sns.barplot(data=df_res, x='class', y='AUPRC', hue='group')
        plt.title(f"AUPRC of {model_name} on the different subgroups of {subgroup_name}")
        ax.set_ylim(0,1)
        
        x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches][:len(df_res)]
        y_coords = [p.get_height() for p in ax.patches][:len(df_res)]
        df_res["CI_AUPRC_low"] = df_res["AUPRC"] - df_res["CI_AUPRC_low"]
        df_res["CI_AUPRC_up"] = df_res["CI_AUPRC_up"] - df_res["AUPRC"]
        yerr = list(np.array([df_res["CI_AUPRC_low"].tolist(),df_res["CI_AUPRC_up"].tolist()]))
        ax.errorbar(x=x_coords, y=y_coords, yerr=yerr, fmt="none", c="k")

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
        df_model["model"] = model_name
        lst_df.append(df_model)
    
    df_results_combined = pd.concat(lst_df, axis=0, ignore_index=True)
    plt.figure(figsize=(10,10))
    ax = sns.barplot(data=df_results_combined, x='model', y='AUC', hue='group')
    plt.title(f"AUC of models on images with and without chest drains")
    ax.set_ylim(0,1)
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches][:len(df_results_combined)]
    y_coords = [p.get_height() for p in ax.patches][:len(df_results_combined)]
    df_results_combined["CI_AUC_low"] = df_results_combined["AUC"] - df_results_combined["CI_AUC_low"]
    df_results_combined["CI_AUC_up"] = df_results_combined["CI_AUC_up"] - df_results_combined["AUC"]
    yerr = list(np.array([df_results_combined["CI_AUC_low"].tolist(),df_results_combined["CI_AUC_up"].tolist()]))
    ax.errorbar(x=x_coords, y=y_coords, yerr=yerr, fmt="none", c="k")
    plt.savefig(f"./reports/figures/subgroups_perf/pneumothorax_drains_auc.png", bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(10,10))
    ax = sns.barplot(data=df_results_combined, x='model', y='AUPRC', hue='group')
    plt.title(f"AUPRC of models on images with and without chest drains")
    ax.set_ylim(0,1)
    
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches][:len(df_results_combined)]
    y_coords = [p.get_height() for p in ax.patches][:len(df_results_combined)]
    df_results_combined["CI_AUPRC_low"] = df_results_combined["AUPRC"] - df_results_combined["CI_AUPRC_low"]
    df_results_combined["CI_AUPRC_up"] = df_results_combined["CI_AUPRC_up"] - df_results_combined["AUPRC"]
    yerr = list(np.array([df_results_combined["CI_AUPRC_low"].tolist(),df_results_combined["CI_AUPRC_up"].tolist()]))
    ax.errorbar(x=x_coords, y=y_coords, yerr=yerr, fmt="none", c="k")
    plt.savefig(f"./reports/figures/subgroups_perf/pneumothorax_drains_auprc.png", bbox_inches='tight', dpi=300)
    plt.close()

def generate_calibration_curves(models):
    plt.figure(figsize=(10,10))
    plt.plot([0, 1], 
         [0, 1], 
         linestyle='--', 
         label='Perfectly Calibrated')
    for model_name in models:
        df_probas = pd.read_csv(f"./data/probas_CXR14/probas_CXR14_{model_name}.csv")
        y_true = df_probas["Pneumothorax"]
        y_proba = df_probas["proba_Pneumothorax"]
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        plt.plot(prob_pred,prob_true,marker='o',linewidth=1,label=models[model_name])
    plt.title('Calibration Curves for all CLIP-based models')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.legend()
    plt.savefig(f"./reports/figures/calibration_cxr14.png", bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
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
    # generate_table_zeroshot_mimic(models,labels,"global")
    # generate_barplot_subgroup(models,"MIMIC",["global","Female","Male"],"sex")
    # generate_barplot_subgroup(models,"MIMIC",["global","White","Black","Asian"],"race")
    # generate_barplot_subgroup(models,"MIMIC",["global","18-25","25-50","50-65","65-80","80+"],"age")
    generate_barplot_drains(models)
    generate_calibration_curves(models)