import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, re
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]

sys.path.append(str(root_dir))


from source import RESULT_DIR, MODEL_DIR, DATA_DIR, PERT_ESC_WIDE, PERT_FLUX_WIDE, PERT_CHANGES_WIDE, UNIPROT_ID_2_DATA, MODEL_ENZYME_2_PMET, ESC_DATA_WIDE
from source.GEM import GEM

### OUTPUT PATHS

EXCEL_ESC_SUMMARY = RESULT_DIR / "enzymes_persistent_dominant_active.xlsx"

MANU_TABLE = RESULT_DIR / "Manscript_table_perturbation.csv"

### DATA PATHS
def main():


    print("== DATA PROPERTIES ==")
    ## COLS: SampleID | Enz1 | Enz2 | ...
    # SampleID is "T[temp]_[sample]"

    samples = pd.read_csv(PERT_ESC_WIDE,index_col=0)
    
    ## Inactive enzymes
    sensitivity_sums = samples.set_index("SampleID").sum(axis=0)
    print(len(sensitivity_sums))
    print(f"Enzymes having zero sensitivity at all temperatures and samples : {sum(sensitivity_sums==0)}")


    dfs = list()
    for temp in np.arange(10,41,1):
        temp_samples = samples.loc[samples["SampleID"].str.contains(f"T{temp}_"),:].drop(columns="SampleID")
        perc_non_zero = (temp_samples != 0).mean() 
        dfs.append(perc_non_zero)

    perc_non_zero = pd.concat(dfs,axis=1)
    perc_non_zero.columns = np.arange(10,41,1)

    print("== GROUPING OF ENZYMES ==")
    # count for groups 
    # 1. non-zero in all runs
    group1 = (perc_non_zero==1).sum(axis=0)
    # 2. non-zero in >=50% of runs
    group2 = ((perc_non_zero<1) & (perc_non_zero>=0.5)).sum(axis=0)
    # 3. non-zero in <50% of runs
    group3 = ((perc_non_zero>0) & (perc_non_zero<0.5)).sum(axis=0)
    # 4. non-zero in no runs
    group4 = (perc_non_zero==0).sum(axis=0)
    
    group_assignments = pd.concat([group1,group2,group3,group4],axis=1)
    group_assignments.columns=["Non-zero in all runs","Non-zero in more than half","Non-zero in less than half","Zero in all runs"]

    group_assignments=group_assignments.reset_index(names="Temperature").melt(id_vars="Temperature",value_name = "Count",var_name = "Group")
    plot_df = group_assignments.pivot_table(index="Temperature", columns="Group", values="Count")


    # Normalize to proportions (each row sums to 1)
    plot_df = plot_df.div(plot_df.sum(axis=1), axis=0)

    # Plot
    fig, ax = plt.subplots()
    ax.stackplot(plot_df.index, plot_df.T, labels=plot_df.columns)
    ax.legend(loc="upper left")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    plt.show()



    print(f"Number of enzymes active at all temperature and samples (consistent) {sum(perc_non_zero.sum(axis=1) == 31)}")
    consistent_enz = perc_non_zero.index[perc_non_zero.sum(axis=1) == 31].tolist()
    print(f"Number of enzymes inactive at all temperature and samples (inactive) {sum(perc_non_zero.sum(axis=1) == 0)}")
    inactive_enz = perc_non_zero.index[perc_non_zero.sum(axis=1) == 0].tolist()

    # Enzymes always active at specific temperatures but not all
    print(f"Number of enzymes always active at specific temperatures but not all (primary limiting): {sum(perc_non_zero.agg(lambda x: (any(x==1) & (sum(x)!=31)),axis=1))}")
    prim_limiting = perc_non_zero.index[perc_non_zero.agg(lambda x: (any(x==1) & (sum(x)!=31)),axis=1)].tolist()
    print(f"Number of enzymes sometimes active at specific temperatures (secondary limiting): {sum(perc_non_zero.agg(lambda x: (all(x<1) & (sum(x)!=0)),axis=1))}")
    sec_limiting = perc_non_zero.index[perc_non_zero.agg(lambda x: (all(x<1) & (sum(x)!=0)),axis=1)].tolist()#

    # Dominant enzymes
    col_max = samples.drop(columns="SampleID").max(axis=0)
    dominant_enz = col_max[col_max>0.1].index


    # Average ESC across samples
    avg_entry_ESC = samples.drop(columns="SampleID").mean(axis=0).squeeze().to_dict()

    # Temperature at highest sensitivity
    max_sens_temp = samples.set_index("SampleID").drop(columns = inactive_enz).idxmax().squeeze().to_dict()


    UniProt = pd.read_csv(UNIPROT_ID_2_DATA, sep="\t")
    model = GEM(MODEL_DIR / "TGEMAdj.mat")
    dicti = {"Enzyme":[], "Active":[], "Persistent":[], "Dominant":[], "Primary limiting":[], "Secondary limiting":[], "Avg. ESC":[], "Max ESC Temp":[], "Protein names":[], "Uniprot genes":[]}

    for enz_ID in model.enzymes:

        dicti["Enzyme"].append(enz_ID)
        if enz_ID not in inactive_enz:
            dicti["Active"].append(1)
            match = re.search(r"(?<=T)[0-9]{2}(?=_)", max_sens_temp[enz_ID])
            dicti["Max ESC Temp"].append(match.group() if match else None)
        else:
            dicti["Active"].append(0)
            dicti["Max ESC Temp"].append("-")
        if enz_ID in consistent_enz:
            dicti["Persistent"].append(1)
        else:
            dicti["Persistent"].append(0)
        if enz_ID in dominant_enz:
            dicti["Dominant"].append(1)
        else:
            dicti["Dominant"].append(0)
        
        if enz_ID in prim_limiting : 
            dicti["Primary limiting"].append(1)
        else:
            dicti["Primary limiting"].append(0)

        if enz_ID in sec_limiting : 
            dicti["Secondary limiting"].append(1)
        else:
            dicti["Secondary limiting"].append(0)

        dicti["Protein names"].append(UniProt[UniProt["Entry"]==enz_ID]["Protein names"].values[0] if enz_ID in UniProt["Entry"].values else np.nan)
        genes = UniProt[UniProt["Entry"]==enz_ID]["Gene Names"].values[0] if enz_ID in UniProt["Entry"].values else ""
        genes = genes.replace(" ",", ")
        dicti["Uniprot genes"].append(genes)
        dicti["Avg. ESC"].append(avg_entry_ESC[enz_ID])

    df_enz = pd.DataFrame(dicti)
    df_enz = df_enz.sort_values(by=["Dominant","Persistent","Primary limiting","Secondary limiting","Active","Avg. ESC","Protein names"], ascending=False)
    col_order = ["Enzyme","Dominant","Persistent","Primary limiting","Secondary limiting","Active","Avg. ESC","Max ESC Temp","Protein names","Uniprot genes"]
    df_enz = df_enz.loc[:,col_order]
    df_enz.to_excel(EXCEL_ESC_SUMMARY, index=False)
    print("Excel summary table saved at:")
    print(str(EXCEL_ESC_SUMMARY))


    pub_table = df_enz.loc[df_enz["Avg. ESC"]>0.01,:]
    rel_cols = ["Enzyme","Dominant","Persistent","Avg. ESC","Max ESC Temp","Uniprot genes"]
    categ_dict = {"Persistent":"At all $s$ and $T$", 
                "Primary limiting": "At all $s$ at some $T$",
                "Secondary limiting":"At some $s$ at some $T$"}

    pub_table.loc[:,"Limiting growth"] = pub_table.loc[:,["Persistent","Primary limiting","Secondary limiting"]].idxmax(axis=1).map(pd.Series(categ_dict))
    pub_table = pub_table.rename({"Enzyme":"UniProt ID", "Uniprot genes":"Genes","Protein names":"Protein name"},axis=1).loc[:,["UniProt ID","Avg. ESC","Max ESC Temp","Limiting growth","Genes","Protein name"]]
    pub_table = pub_table.sort_values(by=["Avg. ESC","Protein name"], ascending=False)
    pub_table["Avg. ESC"] = np.round(pub_table["Avg. ESC"],4)

    pub_table["Protein name"] = pub_table["Protein name"].replace(to_replace=" \(.+",regex=True,value='')

    pub_table.to_csv(MANU_TABLE)    
    print("Publication table saved at:")
    print(str(MANU_TABLE))
    print()


    print("== ISOENZYME PROPORTIONS ==")
    isoenzyme_map = pd.read_csv(MODEL_ENZYME_2_PMET,index_col=0)
    isoenzyme_map = isoenzyme_map.loc[:,["Enzymes","Type"]].drop_duplicates()
    ## Which enzymes are in the Persistent group?

    persistent_enz = df_enz.loc[df_enz["Persistent"]==1,"Enzyme"].tolist()
    ## Isoenzymes
    print("Persistent enzymes")
    isIso = pd.Series(persistent_enz).isin(isoenzyme_map["Enzymes"])
    n = len(persistent_enz)
    n_Iso = sum(isIso)
    n_nonIso = n- n_Iso
    print(n)
    print(f"Non-isoenzymes: {n_nonIso} - {n_nonIso / n * 100}")
    print(f"Isoenzymes: {n_Iso} - {n_Iso / n * 100}")
    print(isoenzyme_map.loc[isoenzyme_map["Enzymes"].isin(persistent_enz),"Type"].value_counts())
    print()

    print("Nonessential enzymes: baseline persistent - perturbation persistent")
    print("Enzyme sensitivity is not robust to perturbation of kcat.")
    SC_wide = pd.read_csv(ESC_DATA_WIDE,index_col=0)
    bl_per_enz = SC_wide.index[SC_wide.min(axis=1)>0]
    non_ess_per_enz = np.setdiff1d(bl_per_enz,persistent_enz)
    isIso = pd.Series(non_ess_per_enz).isin(isoenzyme_map["Enzymes"])
    n = len(non_ess_per_enz)
    n_Iso = sum(isIso)
    n_nonIso = n- n_Iso
    print(n)
    print(f"Non-isoenzymes: {n_nonIso} - {n_nonIso / n * 100}")
    print(f"Isoenzymes: {n_Iso} - {n_Iso / n * 100}")
    print(isoenzyme_map.loc[isoenzyme_map["Enzymes"].isin(non_ess_per_enz),"Type"].value_counts())
    print()

    print("Inactive enzymes")
    isIso = pd.Series(inactive_enz).isin(isoenzyme_map.loc[isoenzyme_map["Type"]!="Obligate","Enzymes"])
    n = len(inactive_enz)
    n_Iso = sum(isIso)
    n_nonIso = n- n_Iso
    print(n)
    print(f"Non-isoenzymes: {n_nonIso} - {n_nonIso / n * 100}")
    print(f"Isoenzymes: {n_Iso} - {n_Iso / n * 100}")
    print(isoenzyme_map.loc[isoenzyme_map["Enzymes"].isin(inactive_enz),"Type"].value_counts())
    print()

    print("Isoenzymes")
    print(len(isoenzyme_map.loc[isoenzyme_map["Type"]!="Obligate","Enzymes"].unique()))
    print()



if __name__ == "__main__":
    main() 