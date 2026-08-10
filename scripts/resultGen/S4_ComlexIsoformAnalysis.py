import sys, os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))
from source import SUPP_FIG_DIR, PROTEOMICS_DATA, PERT_ESC_WIDE, MODEL_ID_NAME_MAP


subunits = {"RuBisCO":["O03042",
                    "P10795",
                    "P10796",
                    "P10797",
                    "P10798"],
            "TrpS": ["O22765",
                    "Q42529",
                    "P14671",
                    "P25269"],
            "Cytb6":["Q9ZR03",
                    "P56775",
                    "P56776",
                    "O48717",
                    "P61039",
                    "P56774",
                    "P56771",
                    "P56773"],
            "bCA" : ["P27140",
                    "P42737",
                    "Q9ZUC2",
                    "Q94CE4",
                    "Q94CE3",
                    "Q9C6F5"],
            "CA": ['F4JIK2', 
                   'Q8L817', 
                   'O04846', 
                   'Q94CE4',
                   'Q9C6F5', 
                   'P27140', 
                   'Q94CE3',
                   'F4HUC4', 
                   'Q9FYE3', 
                   'Q9SUB4', 
                   'F4IHR4', 
                   'Q9ZUC2', 
                   'P42737'],
            "ATPS":['P09468', 
                    'P19366',
                    'P56757',
                    'P56758',
                    'P56759',
                    'P56760',
                    'Q01908',
                    'Q9SSS9',
                    'Q01909']
                        }

complex_names = {"RuBisCO":"RuBisCO",
                 "TrpS": "Tryptophan synthase",
                 "Cytb6":"Cytochrome b6-f",
                 "bCA":"Beta carbonic anhydrases",
                 "CA":"Carbonic anhydrases",
                 "ATPS": "ATP synthase"}


def main():
    enz_name = pd.read_csv(MODEL_ID_NAME_MAP,index_col=0).loc[:,"Name"].squeeze().to_dict()
    # Measured abundance data
    acc = ["Bur.0","Pla.0"]
    mabund = pd.read_csv(PROTEOMICS_DATA,index_col=0)
    mabund = mabund[mabund["Accession"].isin(acc)]

    # Sensitivity data
    '''
    temp = [17,27]
    full_data = pd.read_csv(PERTURBATION_DATA,index_col=0)
    temp_data = full_data[full_data["Temperature"].isin(temp)]
    temp_data.drop(columns=["Type","RunID"],inplace=True)
    temp_data.loc[:,"ModelID"] = temp_data["ModelID"].str.removeprefix("draw_prot_")
    '''
    temp = [17,27]
    full_data = pd.read_csv(PERT_ESC_WIDE,index_col=0)
    extracted = full_data["SampleID"].str.extract(r"T(\d+)_(\d+)")
    full_data["Temperature"] = extracted[0].astype(int)
    full_data["RunID"] = extracted[1].astype(int)
    temp_data = full_data[full_data["Temperature"].isin(temp)].drop(columns=["SampleID"])
    temp_data = temp_data.melt(id_vars=["Temperature","RunID"],var_name="ModelID",value_name="Sensitivity Index").reset_index()


    if not os.path.isdir(SUPP_FIG_DIR / "ComplexAbundanceComparison"):
        os.mkdir(SUPP_FIG_DIR / "ComplexAbundanceComparison")


    for comp, enzymes in subunits.items():
        color_map = {enz: color for enz, color in zip(enzymes, sns.color_palette("husl", len(enzymes)))}


        cut_mabund = mabund[mabund["UniprotID"].isin(enzymes)]
        covered_enzymes = cut_mabund["UniprotID"].unique()
        print(f"Measurements cover {len(covered_enzymes)} of the enzymes in complex {comp}.")
        cut_sens = temp_data.loc[temp_data["ModelID"].isin(enzymes),:]

        fig, axes = plt.subplots(nrows=3,figsize = (6,8),sharex=True,constrained_layout= True)

        i=0
        for acc in ["Bur.0","Pla.0"]:
            t = cut_mabund[cut_mabund["Accession"]==acc]
            t = pad_missing_hue_levels(t, "Temperature", "UniprotID", enzymes)
            sns.boxplot(t,x="Temperature",y="Relative abundance",hue="UniprotID",hue_order = enzymes,palette=color_map,ax=axes[i])
            axes[i].set_ylabel(f"Measured abundance {acc.replace(".","-")}")
            axes[i].legend_.remove()
            i+=1

        sns.barplot(cut_sens,x="Temperature",y="Sensitivity Index",hue="ModelID",hue_order = enzymes,palette=color_map,errorbar=("ci",95),ax=axes[2],estimator="mean")
        handles, labels = axes[2].get_legend_handles_labels()
        axes[2].legend_.remove()
        new_labels = [enz_name[l] if enz_name[l] != "-" else l for l in labels]

        fig.legend(handles,new_labels,ncol=3,title="Gene",loc="upper center",bbox_to_anchor=(0.5,0))
        fig.savefig(SUPP_FIG_DIR /f"ComplexAbundanceComparison/{comp}_abundance_comparison.png",bbox_inches="tight")
    print("Plots were saved at:")
    print(str(SUPP_FIG_DIR) + f"ComplexAbundanceComparison/")

def pad_missing_hue_levels(df, x_col, hue_col, hue_levels):
    """Ensure every (x, hue) combination exists so seaborn dodge-width
    calculations always divide by len(hue_levels), even when some
    combinations have no real data."""
    x_vals = df[x_col].unique()
    existing = set(map(tuple, df[[x_col, hue_col]].drop_duplicates().values))
    full_combos = {(x, h) for x in x_vals for h in hue_levels}
    missing = full_combos - existing

    if missing:
        placeholder = pd.DataFrame(missing, columns=[x_col, hue_col])
        df = pd.concat([df, placeholder], ignore_index=True)

    df[hue_col] = pd.Categorical(df[hue_col], categories=hue_levels, ordered=True)
    return df


if __name__ == "__main__":
    main()