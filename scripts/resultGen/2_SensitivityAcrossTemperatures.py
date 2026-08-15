import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, io
from pathlib import Path
import scipy.stats as sp
from statsmodels.stats import multitest

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))
from source import MODEL_DIR, MODEL_ENZYME_2_PMET, ESC_DATA_WIDE ,PFBA_DATA, MODEL_ENZYME_2_SUBSYSTEM, FLUX_SAMP_WIDE, SUPP_RES_DIR, SUPP_FIG_DIR
from source.GEM import GEM

class silence():
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr

        self.text_trap = io.StringIO()
        sys.stdout = self.text_trap
        sys.stderr = self.text_trap
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr


def main():

    print("== GENERAL CLASSIFICATION ==")

    ### Count active enzymes
    SC_wide = pd.read_csv(ESC_DATA_WIDE,index_col=0)
    active_enzyme = SC_wide.index[SC_wide.sum(axis=1)>0]
    inactive_enzyme = SC_wide.index[SC_wide.sum(axis=1)==0]
    PLE = SC_wide.index[SC_wide.all(axis=1)]
    OLE = np.setdiff1d(active_enzyme,PLE)
    print(f"\tActive enyzmes (ESC > 0 at any temperature): {len(active_enzyme)}")
    print(f"\tInactive enyzmes (ESC = 0 at all temperatures): {len(inactive_enzyme)}")
    print(f"\tPersistently limiting enzymes - PLE (ESC > 0 at all temperatures): {len(PLE)}")
    print(f"\tOcassionally limiting enzymes - OLE (ESC > 0 at some temperatures): {len(OLE)}")
    print()
    active_enzme_at_temp= [sum(SC_wide.loc[:,c]!=0) for c in SC_wide.columns]
    print(f"\tHighest number of LE at {np.argmax(active_enzme_at_temp)+10}°C with {np.max(active_enzme_at_temp)}")
    print(f"\tLowest number of LE at {np.argmin(active_enzme_at_temp)+10}°C with {np.min(active_enzme_at_temp)}")
    print()

    # Correlation between temperature and number of non-zero ESC
    print("\tSpearman correlation between number of LE and Temperature in °C:")
    print(f"\t{sp.spearmanr(active_enzme_at_temp,np.arange(10,41))}")
    print()

    print("==TEMPERATURE DEPENDENCY OF METABOLIC FLUX DISTRIBUTIONS==")
    print("H1: Is there a qualitative change in metabolic flux distributions across temperatures?")
    # Correlation between the median sampled fluxes at each temperature
    Flux_sampling_data = pd.read_csv(FLUX_SAMP_WIDE,index_col=0)
    isorxn_map = pd.read_csv(MODEL_ENZYME_2_PMET)
    data = Flux_sampling_data.groupby("Temperature").median().drop(columns="SampleID")
    data = data.drop(columns=isorxn_map["Isorxns"].unique()) # Remove isoreactions, as changes in catalyzing enzyme do not necessitate changes in metabolic flux distribution. 
    data = data.drop(columns = data.columns[data.columns.str.contains("prot_")]) # drop enzyme abundances, same reason as above.
    cor_mat_spear = data.transpose().corr(method="spearman").to_numpy()
    cor_mat_pear = data.transpose().corr(method="pearson").to_numpy()
    print("Sampling data:")
    print(f"Avg. pearson correlation of fluxes between temperatures: {np.mean(cor_mat_pear[np.triu_indices(31)])}±{sp.sem(cor_mat_pear[np.triu_indices(31)])}, std: {np.std(cor_mat_pear[np.triu_indices(31)])}")
    print(f"Avg. spearman correlation of fluxes between temperatures: {np.mean(cor_mat_spear[np.triu_indices(31)])}±{sp.sem(cor_mat_spear[np.triu_indices(31)])}, std: {np.std(cor_mat_spear[np.triu_indices(31)])}")
    print()

    # Correlation between the pFBA flux solution at each temperature
    pFBA_df = pd.read_csv(PFBA_DATA, index_col=0)
    data = pFBA_df.pivot(columns="Temperature",index="Variable",values="Flux").transpose()
    data = data.drop(columns=isorxn_map["Isorxns"].unique())
    data = data.drop(columns = data.columns[data.columns.str.contains("prot_")])
    cor_mat_spear = data.transpose().corr(method="spearman").to_numpy()
    cor_mat_pear = data.transpose().corr(method="pearson").to_numpy()
    print("pFBA data:")
    print(f"Avg. pearson correlation of fluxes between temperatures: {np.mean(cor_mat_pear[np.triu_indices(31)])}±{sp.sem(cor_mat_pear[np.triu_indices(31)])}, std: {np.std(cor_mat_pear[np.triu_indices(31)])}")
    print(f"Avg. spearman correlation of fluxes between temperatures: {np.mean(cor_mat_spear[np.triu_indices(31)])}±{sp.sem(cor_mat_spear[np.triu_indices(31)])}, std: {np.std(cor_mat_spear[np.triu_indices(31)])}")
    print()

    print("H2: Are observed changes in number of LE solely due to isoenzyme substitutions")
    # How many isoenzymes are in OLE set?
    non_obligate_enz = isorxn_map.loc[isorxn_map["Type"]!="Obligate","Enzymes"]
    print(f"Isoenzymes in OLE set: {len(np.intersect1d(OLE,non_obligate_enz))}/{len(OLE)})")
    print()

    # How many arm reactions carry flux persistently?
    #   Summarize the relation btw. enzyme and arm reactions
    model = GEM(MODEL_DIR / f"TGEMAdj_20.mat")
    inter = np.intersect1d(OLE,non_obligate_enz)
    dicti = {"Enzyme":[],"Reaction":[],"ArmRxn":[]}
    for enz in inter:
        rxns = model.get_catalyzed_rxns(enz)
        for rxn in rxns:
            with silence():
                arm_rxn = model.get_arm_reaction(rxn)
            dicti["Enzyme"].append(enz)
            dicti["Reaction"].append(rxn)
            dicti["ArmRxn"].append(arm_rxn)

    #  Map Arm reactions, using non-arm reactions where there are none.
    temp = pd.DataFrame(dicti)
    rxns = temp["ArmRxn"]
    rxns[rxns.isna()] = temp.loc[rxns.isna(),"Reaction"]
    temp.loc[:,"MapRxn"] = rxns

    
    print(f"These catalyze {len(temp["Reaction"].unique())} reactions")
    print(f"These catalyze {len(temp["ArmRxn"].unique())} arm reactions")

    # Use the median of the flux sampling as a proxy for flux at each temperature. 
    #   Fluxes carrying less than 100th of their maximum flux across temperatures are deemed inactive (=100).
    data = Flux_sampling_data.groupby("Temperature").median().drop(columns="SampleID").loc[:,rxns].transpose().reset_index(names="MapRxn")
    # enzyme | reaction | arm reaction | map rxn | sampling data 
    arm_df = temp.merge(data,on="MapRxn",how="left").drop_duplicates()

    
   
    # enzyme | reaction | map rxn | sampling data 
    df = arm_df.drop(columns="ArmRxn")
    # map rxn | sampling data 
    MapRxn_df = df.drop(columns=["Enzyme","Reaction"]).drop_duplicates().set_index("MapRxn")
    # Enzyme | sampling data 
    #Enz_df = df.drop(columns=["MapRxn","Reaction"]).drop_duplicates().set_index("Enzyme")


    persistent_arm = MapRxn_df.index[(MapRxn_df.min(axis=1) > (MapRxn_df.max(axis=1)/100))]
    non_persistent_arm =  MapRxn_df.index[(MapRxn_df.min(axis=1) <= (MapRxn_df.max(axis=1)/100))]
    #persistent_OLE =  Enz_df.index[(Enz_df.min(axis=1) > (Enz_df.max(axis=1)/100))]
    #non_persisten_OLE = Enz_df.index[(Enz_df.min(axis=1) <= (Enz_df.max(axis=1)/100))]
    
    print(f"{len(persistent_arm)} arm reactions carry flux at every temperature")
    print(f"{len(non_persistent_arm)} arm reactions do not carry flux at every temperature")

    per_arm_OLE = temp.loc[temp["ArmRxn"].isin(persistent_arm),"Enzyme"].unique()
    print(f"{len(per_arm_OLE)} OLE have arm reactions that carry flux at every temperature")
    print()

    print("== SUBSYSTEM ENRICHMENT ==")
    # Subsystem enrichment for each group
    subs_map = pd.read_csv(MODEL_ENZYME_2_SUBSYSTEM,index_col=0)
    minimum_enzymes_in_subsystem = 5
    
    # hypergeometric test 
    # N = 671 All enzymes
    # K = enzymes in subsystem
    # n = enzymes in group
    # k = enzymes in group and subsystem
    print(len(inactive_enzyme))
    print(len(active_enzyme))
    
    removed_subsystems = []
    dicti = {"Group":[],"Subsystem":[],"Statistic":[],"P-value":[]}
    for key, enz_oi in {"Active":active_enzyme,
                        "PLE":PLE,
                        "OLE":OLE,
                        "tdOLE":df["Enzyme"].unique()}.items():
            for sub in subs_map["Subsystem"].unique():
                    enz_in_sub = subs_map.loc[subs_map["Subsystem"]==sub,"Enzyme"].unique()
                    N = 671
                    K = len(enz_in_sub)
                    if(K<minimum_enzymes_in_subsystem):
                            removed_subsystems.append(sub)
                            continue
                    n = len(enz_oi)
                    k = len(np.intersect1d(enz_in_sub,enz_oi))
                    ctab = [[k , n-k],
                            [K-k, N-K-n+k]]
                    res = sp.fisher_exact(ctab, alternative='greater')
                    dicti["Group"].append(key)
                    dicti["Subsystem"].append(sub)
                    dicti["Statistic"].append(res.statistic)
                    dicti["P-value"].append(res.pvalue)
    print(f"Subsystems removed because they contained less than {minimum_enzymes_in_subsystem} enzymes:")
    print(set(removed_subsystems))
    enrichment_df = pd.DataFrame(dicti)
    enrichment_df_wide = enrichment_df.pivot(columns="Group",values="P-value",index="Subsystem")
    for i in np.arange(4):
            enrichment_df_wide.iloc[:,i]=multitest.multipletests(enrichment_df_wide.iloc[:,i],alpha=0.05,method="fdr_bh")[1]

    enrichment_df_wide.to_csv(SUPP_RES_DIR / "LE_sets_enrichment_results.csv")
    print(f"Enrichment results are saved at:")
    print(f"{str(SUPP_RES_DIR) + '/LE_sets_enrichment_results.csv'}")
    print()

    print("== CHANGES IN LE SETS ACROSS TEMPERATURES ==")
    ## Jaccard between temperatures
    jac_mat = np.zeros((31,31))
    temp_range = np.arange(10,41)
    i=0
    for temp in temp_range:
        set1 = SC_wide.loc[SC_wide.loc[:,str(temp)]!=0,:].index
        j=0
        for temp2 in temp_range:
            set2 = SC_wide.loc[SC_wide.loc[:,str(temp2)]!=0,:].index
            jac_mat[i,j]=len(np.intersect1d(set1,set2)) / len(np.union1d(set1,set2))
            j+=1
        i+=1

    # Jaccard summary as a function of distance
    dist_df = pd.DataFrame({"Distance":np.arange(1,31)})
    mean_vec = np.zeros((30,))
    std_vec = np.zeros((30,))
    for dist in np.arange(1,31):
        mean_vec[dist-1] = np.mean(np.diag(jac_mat,k=dist))
        std_vec[dist-1] = np.std(np.diag(jac_mat,k=dist))
    dist_df["Mean jaccard"] = mean_vec
    dist_df["Std jaccard"] = std_vec
    dist_df.to_csv(SUPP_RES_DIR / "LE_set_jaccard_per_distance.csv")
    print(f"LE set Jaccard by temperature distance data is saved at:")
    print(f"{str(SUPP_RES_DIR) + "/LE_set_jaccard_per_distance.csv"}")
    print(f"Jaccard similarity at $\\Delta$T = 1 : {dist_df.iloc[0,1]}±{dist_df.iloc[0,2]}")
    print(f"Jaccard similarity at $\\Delta$T = 29 : {dist_df.iloc[28,1]}±{dist_df.iloc[28,2]}")
    print()

    diff_vec = np.diag(jac_mat,k=1)
    temps = np.arange(10,41)
    changes = temps[:-1].astype(str) + " to " + temps[1:].astype(str) 
    jac_sim_df= pd.DataFrame({"Change":changes,
                            "Jaccard similarity":diff_vec})
    threshold = 0.97
    print(f"Consecutive temperatures with a jaccard similarity above {threshold}:")
    print(jac_sim_df.loc[jac_sim_df["Jaccard similarity"]>threshold,"Change"])
    print()
    # Sets of limiting enzymes at each temperature
    LE_sets = dict()
    for temp in temp_range:
        set1 = SC_wide.loc[SC_wide.loc[:,str(temp)]!=0,:].index
        LE_sets[temp] = set1

    # Block intersections
    
    blocks = {"A":[12,15],
            "B":[16,19],
            "C":[20,23],
            "D":[24,26]}
    print("Blocks of consecutive temperatures with high jaccard distance:")
    for id, temp in blocks.items():
         print(f"{id}: {temp[0]}°C to {temp[1]}°C")
    print()

    block_intersects = dict()
    for block, temprange in blocks.items():
        temps = np.arange(temprange[0],temprange[1]+1)
        nf=True,
        LE_inter=[]
        for temp in temps:
            if(nf):
                LE_inter = LE_sets[temp]
                nf=False
            else:
                LE_inter = np.intersect1d(LE_inter,LE_sets[temp])
        block_intersects[block]=LE_inter

    # Symmetric difference between blocks
    block_names = ["A","B","C","D"]
    set_differences = dict()

    for i in np.arange(0,4):
        block1_LE = block_intersects[block_names[i]]
        for j in np.arange(0,4):
            if i==j:
                continue
            else:
                block2_LE = block_intersects[block_names[j]]
                set_differences[f"{block_names[i]}!{block_names[j]}"] = np.setdiff1d(block1_LE,block2_LE)
                set_differences[f"{block_names[j]}!{block_names[i]}"] = np.setdiff1d(block2_LE,block1_LE)
                

    sym_differences = dict()
    for i in np.arange(0,3):
            block1_LE = block_intersects[block_names[i]]
            for j in np.arange(i+1,4):
                block2_LE = block_intersects[block_names[j]]
                sym_differences[f"{block_names[i]}!{block_names[j]}"] = np.union1d(np.setdiff1d(block1_LE,block2_LE),np.setdiff1d(block2_LE,block1_LE))

    # Print the subsystems that are represented in the symmetric difference between blocks.
    for block_comp, symdiff in sym_differences.items():
        temp = subs_map[subs_map["Enzyme"].isin(symdiff)]
        print(f"Symmetric difference {block_comp}:")
        print(f"Size: {len(symdiff)}")
        print(temp["Subsystem"].value_counts().head())
        print()


    fig, ax = plt.subplots(ncols=2,figsize=(13,5))
    sns.lineplot(data=dist_df,x="Distance",y="Mean jaccard",errorbar=None,ax = ax[1])
    ax[1].errorbar(
        x=dist_df["Distance"],
        y=dist_df["Mean jaccard"],
        yerr=dist_df["Std jaccard"],
        fmt="none",      # no marker
        capsize=3,
        color="black",
        alpha=0.7
    )
    ax[1].set_xlabel("$\\Delta T$ [°C]")
    ax[1].set_ylabel("Mean Jaccard similarity")
    ax[1].set_ylim(0.6,1)

    jac_mat
    sns.heatmap(pd.DataFrame(jac_mat,columns=np.arange(10,41),index = np.arange(10,41)),ax=ax[0])
    ax[0].set_xlabel("Temperature [°C]")
    ax[0].set_ylabel("Temperature [°C]")
    fig.text(0.1,0.9,s="A",fontdict={"weight":"bold","size":14})
    fig.text(0.5,0.9,s="B",fontdict={"weight":"bold","size":14})
    fig.savefig(SUPP_FIG_DIR / "Jaccard_similarity_between_temperatures.png",dpi=300,bbox_inches="tight")

    print(f"A heatmap of the jaccard distances of LE sets at different temperatures was generated at:")
    print(f"{str(SUPP_FIG_DIR) + "/Jaccard_similarity_between_temperatures.png"}")



if __name__ == "__main__":
    main()