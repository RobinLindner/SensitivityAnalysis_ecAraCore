import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, re
from pathlib import Path
from matplotlib.colors import LogNorm, TwoSlopeNorm
from statsmodels.stats import multitest
import scipy.stats as sp

root_dir = Path(__file__).resolve().parents[2]

sys.path.append(str(root_dir))

from source import SUPP_FIG_DIR, MODEL_ENZYME_2_SUBSYSTEM, ESC_DATA_WIDE

def main():

    SC_wide = pd.read_csv(ESC_DATA_WIDE,index_col=0)
    subs_map = pd.read_csv(MODEL_ENZYME_2_SUBSYSTEM,index_col=0)
    minimum_enzymes_in_subsystem = 5
    # Subsystem enrichment for each group
    # hypergeometric test 
    # N = 671 All enzymes
    # K = enzymes in pathway
    # n = enzymes in group
    # k = enzymes in group and pathway
    removed_subsystems = []
    dicti = {"Temperature":[],"Subsystem":[],"Statistic":[],"P-value":[]}
    for temp in np.arange(10,41,1):
        group_oi = SC_wide[SC_wide.loc[:,str(temp)]>0].index.tolist()
        for sub in subs_map["Subsystem"].unique():
            enz_in_sub = subs_map.loc[subs_map["Subsystem"]==sub,"Enzyme"].unique()
            N = 671
            K = len(enz_in_sub)
            if(K<minimum_enzymes_in_subsystem):
                    removed_subsystems.append(sub)
                    continue
            n = len(group_oi)
            k = len(np.intersect1d(enz_in_sub,group_oi))
            ctab = [[k , n-k],
                    [K-k, N-K-n+k]]
            res = sp.fisher_exact(ctab, alternative = "greater")
            dicti["Temperature"].append(temp)
            dicti["Subsystem"].append(sub)
            dicti["Statistic"].append(res.statistic)
            dicti["P-value"].append(res.pvalue)
    print(f"Subsystems removed because they contained less than {minimum_enzymes_in_subsystem} enzymes:")
    print(set(removed_subsystems))

    norm = TwoSlopeNorm(vcenter=0.05)

    fig,axes = plt.subplots(figsize=(8,4))
    test_df_long = pd.DataFrame(dicti)
    test_df_long["-log10(P-value)"] = -1 * np.log(test_df_long["P-value"])
    test_df_long["BF_adj_p"] = np.clip(test_df_long["P-value"] * 31, min=0,max=1)
    test_df_long["FDR_adj_p"] = multitest.multipletests(test_df_long["P-value"],alpha=0.05,method="fdr_bh")[1]
    plot_df= test_df_long.pivot(index="Subsystem",columns="Temperature",values="FDR_adj_p")
    order = plot_df.min(axis=1).sort_values()
    order = order[order<=0.5]
    plot_df = plot_df.loc[order.index,:]
    plot_df.index = plot_df.index.str.capitalize()
    sns.heatmap(plot_df,cmap="RdBu", norm=norm,
                cbar_kws={'ticks':[0,0.01,0.03,0.05,0.2,0.4],'label':'FDR adjusted p-value'},
                linewidths=0.003, linecolor='black',
                ax = axes)
    #axes.set_title("Enrichment of limiting enzymes in metabolic pathways.")
    axes.set_ylabel("")
    #axes.set_xticklabels([10,15,20,25,30,35,40])
    fig.savefig(SUPP_FIG_DIR / "subsystem_enrichment_heatmap.png",dpi=300,bbox_inches = 'tight')
    print("Figure saved at:")
    print(SUPP_FIG_DIR / "subsystem_enrichment_heatmap.png")
    plt.show()
    fig.clear()

    print("== ENZYMES PER SUBSYSTEM ==")
    print(f"The model contains {len(subs_map["Subsystem"].unique())} subsystems")
    print()
    

    dicti = {"Subsystem":[],"Number of enzymes":[]}
    for sub in subs_map["Subsystem"].unique():
        enz_in_sub = subs_map.loc[subs_map["Subsystem"]==sub,"Enzyme"].unique()
        dicti["Subsystem"].append(sub)
        dicti["Number of enzymes"].append(len(enz_in_sub))

 
    plot_df = pd.DataFrame(dicti)
    sns.histplot(plot_df,x="Number of enzymes",bins=50)
    plt.title( "Size of subsystems")
    plt.show()


if __name__ == "__main__":
    main()