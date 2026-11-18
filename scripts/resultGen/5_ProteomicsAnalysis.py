import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
import statsmodels.formula.api as smf
import statsmodels.genmod.families.family as fam
from scipy.special import logit
import matplotlib.pyplot as plt
import scipy.stats as st
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import warnings
warnings.filterwarnings("ignore")


root_dir = Path(__file__).resolve().parents[2]

sys.path.append(str(root_dir))

from source import FIG_DIR, ESC_DATA_WIDE, FLUX_SAMP_WIDE, PROTEOMICS_DATA, PERT_ESC_WIDE, MODEL_ENZYME_2_SUBSYSTEM, MODEL_ENZYME_2_PMET, SUPP_FIG_DIR, PROT_LONG,LIMITING_CLASS_TAB, CLUST_RES_DIR, LOOKUP_TABLES

def main():
    ### Data prep
    accessions = ["Bur.0","Pla.0"]
    foundation, full_prot_data = createFoundationalDataFrame()

    category_map = pd.read_csv(LOOKUP_TABLES / "subsystem_broad_classification.csv",index_col=0).set_index("index").squeeze().to_dict()

    ### ==== FIGURE 3 - SUMMARY OF ANALYSES ====
    palette = {"Bur.0_neg": "#08519C",
            "Bur.0": "#6BAED6",
            "Bur.0_pos": "#C6DBEF",
            "Pla.0_neg": "#A63603",
            "Pla.0": "#FD8D3C",
            "Pla.0_pos": "#FDD0A2"
            }

    fig = plt.figure(figsize=(15,10),constrained_layout =True)
    gs = GridSpec(2, 2, figure=fig)

    ax_tl = fig.add_subplot(gs[0, 0])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_br = fig.add_subplot(gs[1, 1])

    gs_tr = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.1)
    ax_tr1 = fig.add_subplot(gs_tr[0, 0])
    ax_tr2 = fig.add_subplot(gs_tr[0, 1],sharey=ax_tr1)

    ## Panel 1 - In model to out of model comparison
    plotIn_Outof_model_comparison(foundation, full_prot_data,palette,cax=ax_tl)

    ## Panel 2 - Barplot of norm differences between clusters
    plotNormDiff_clusters(foundation,ax1 = ax_tr1,ax2 = ax_tr2)

    ## Panel 3 - Regplot between fold changes 
    plotRegFC(foundation, palette, ax_bl)
    
    ## Panel 4 - Barplot of correlation coefficients for specific subsystems
    plotSubsystem_correlations(foundation,category_map,palette,cax=ax_br)
    
    # Figure appearance
    handles,labels = ax_tl.get_legend_handles_labels()

    for ax in [ax_tl,ax_bl,ax_br,ax_tr1,ax_tr2]:
        if(ax.legend_):
            ax.legend_.remove()
            
    fig.text(0,1,"A",fontweight="bold",fontsize=14)
    fig.text(0.5,1,"B",fontweight="bold",fontsize=14)
    fig.text(0,0.55,"C",fontweight="bold",fontsize=14)
    fig.text(0.5,0.55,"D",fontweight="bold",fontsize=14)

    ax_tl.grid(True)
    ax_bl.grid(True)

    labels = [l.replace(".","-") for l in labels]
    fig.legend(handles,labels,bbox_to_anchor = (0.5,0),loc="upper center",ncols=2)

    fig.tight_layout()

    fig.savefig(FIG_DIR / "proteomics_analysis_better_subsystems.png",bbox_inches = "tight")


    print()
    ### ==== IN/OUT OF MODEL PROTEIN COMPARISON ====
    print("==== IN/OUT OF MODEL PROTEIN COMPARISON ====")
    pgs_in_model = foundation.ProteinGroup.unique()

    test_data = (full_prot_data
    .loc[:,["PG.ProteinGroups","Accession","Temperature","Relative abundance"]]
    .groupby(["PG.ProteinGroups","Accession","Temperature"])
    .mean()
    .reset_index()
    .pivot(index=["PG.ProteinGroups","Accession"],columns="Temperature",values="Relative abundance")
    .assign(norm_diff = lambda x: (x[27]-x[17])/((x[27]+x[17])/2))
    .reset_index()
    .drop_duplicates()
    )
    test_data["In model"] = False
    test_data.loc[test_data["PG.ProteinGroups"].isin(pgs_in_model),"In model"] = True

    print(test_data.loc[:,["PG.ProteinGroups","In model"]].drop_duplicates().value_counts("In model"))

    def stat_diff(g):
        a , b = g.loc[g["In model"],"norm_diff"] , g.loc[~g["In model"],"norm_diff"] 
        res = st.ttest_ind(a,b,nan_policy="omit",alternative="greater")
        return pd.Series({
            "statistic" : res.statistic,
            "pvalue" : res.pvalue
        })
        
    sig_df = (test_data
            .groupby("Accession")
            .apply(stat_diff,include_groups=False))
    
    print("Two sample t-tests for abundance distributions of proteins mapping to the model and those that do not.")
    print(sig_df)

    adj_pval = st.false_discovery_control(foundation["rel_abu_diff_pval"], method='bh')

    print()
    ### ==== PROTEIN ABUNDANCE DISTRIBUTION TESTS BETWEEN NSRN CLUSTERS ==== 
    print("==== PROTEIN ABUNDANCE DISTRIBUTION TESTS BETWEEN NSRN CLUSTERS ====")
    for sig in [False,True]:
        if(sig):
            filt_foundation = foundation.loc[adj_pval<0.05]
        else:
            filt_foundation = foundation    

        all_changes_df = (filt_foundation
                    .assign(norm_abu_diff = lambda x: (x["mean_rel_abu_27"]-x["mean_rel_abu_17"]) / ((x["mean_rel_abu_27"]+x["mean_rel_abu_17"])/2))
                    .loc[:,["Accession","ProteinGroup","UniprotID","Cluster","norm_abu_diff"]]
                    .drop_duplicates()
                    )

        # Between clusters across accessions
        general_res = comp_cluster_differences(all_changes_df)

        # Between clusters per accession 
        Bur0_res = comp_cluster_differences(all_changes_df[all_changes_df["Accession"]=="Bur.0"])
        Pla0_res = comp_cluster_differences(all_changes_df[all_changes_df["Accession"]=="Pla.0"])

        # For each cluster between accessions
        comp_res = comp_acc_differences(all_changes_df)

        print(f"TWO SAMPLE T-TEST OF PROTEIN ABUNDANCE DISTRIBUTIONS {"FOR SIGNIFICANTLY RESPONSIVE PROTEINS" if sig else ""}")
        print("Between clusters across accessions")
        print(general_res)

        print("Between clusters per accession")
        print("Bur-0")
        print(Bur0_res)
        print("Pla-0")
        print(Pla0_res)

        print("For each cluster between accessions")
        print(comp_res)

    print()
    ### ==== DISTRIBUTION OF FOLD CHANGES ====
    print("==== DISTRIBUTION OF FOLD CHANGES ====")
    print("Median and one-sample t-test in fold changes to test popmean equal to 1")
    ## In proteins mapping to model
    t = (foundation
        .loc[:,["Accession","UniprotID","sens_mean_fold_change","mean_rel_abu_fold_change","LimitingClass"]]
        .replace([0,np.inf],np.nan)
        .dropna()
        .drop_duplicates()
        
    )
    print("Proteins mapped to model")
    print(np.mean(t["mean_rel_abu_fold_change"]))
    print(st.ttest_1samp(t["mean_rel_abu_fold_change"],popmean=1,alternative="less"))
    print()
    ## In all protein
    t = (full_prot_data
    .loc[:,["PG.ProteinGroups","Accession","Temperature","Relative abundance"]]
    .drop_duplicates()
    .groupby(["PG.ProteinGroups","Accession","Temperature"])
    .mean()
    .reset_index()
    .pivot(index=["PG.ProteinGroups","Accession"],columns="Temperature",values="Relative abundance")
    .assign(fold_changes = lambda x: x[27]/x[17])
    .replace([0,np.inf],np.nan)
    .dropna()
    .reset_index()
    )
    print("All protein")
    print(np.mean(t["fold_changes"]))
    print(st.ttest_1samp(t["fold_changes"],popmean=1,alternative="less"))
    print()

    ### ==== CORRELATION OF FOLD CHANGES ====
    print("==== CORRELATION OF FOLD CHANGES ====")
    for sig in [False,True]:
        if(sig):
            filt_foundation = foundation.loc[adj_pval<0.05]
        else:
            filt_foundation = foundation  
        print(f"Correlation of {"significant" if sig else ""} fold changes in measured abudance and sensitivity between 17°C and 27°C")
        ## All proteins
        test_cur = (filt_foundation
        .loc[:,["UniprotID","Accession","mean_rel_abu_17","mean_rel_abu_27","sens_17_mean","sens_27_mean","LimitingClass"]]
        .rename(columns =  {"sens_17_mean" : "Sensitivity_17",
                        "sens_27_mean" : "Sensitivity_27",
                        "mean_rel_abu_17" : "RelAbundance_17",
                        "mean_rel_abu_27" : "RelAbundance_27"})  
        .assign(sens_fold_change = lambda x: x["Sensitivity_27"] / x["Sensitivity_17"],
                abu_fold_change = lambda x: x["RelAbundance_27"] / x["RelAbundance_17"])
        .replace([0,np.inf],np.nan)
        .dropna()
        .drop_duplicates()
        )
        test_cur["log_sens_fold_change"]=np.log10(test_cur["sens_fold_change"])

        # For both accessions
        full_res = comp_fc_corr(test_cur)   

        # For each accession
        Bur0_res = comp_fc_corr(test_cur[test_cur["Accession"]=="Bur.0"])

        Pla0_res = comp_fc_corr(test_cur[test_cur["Accession"]=="Pla.0"])


        print("Both accessions")
        print(full_res)

        print("Bur-0")
        print(Bur0_res)

        print("Pla-0")
        print(Pla0_res)
        print()

    print()
    ### ==== SUBSYSTEM SPECIFIC CORRELATION ====
    print("==== SUBSYSTEM SPECIFIC CORRELATION ====")
    for sig in [False,True]:
        if(sig):
            filt_foundation = foundation.loc[adj_pval<0.05]
        else:
            filt_foundation = foundation    
        df_full = (filt_foundation
        .loc[:,["UniprotID","Accession","Subsystem","sens_17_mean","sens_27_mean","mean_rel_abu_17","mean_rel_abu_27"]]
        .rename(columns =  {"sens_17_mean" : "Sensitivity_17",
                        "sens_27_mean" : "Sensitivity_27",
                        "mean_rel_abu_17" : "RelAbundance_17",
                        "mean_rel_abu_27" : "RelAbundance_27"})
        .drop_duplicates()
        .replace(0,np.nan)
        .dropna()
        .assign(abu_fold_changes = lambda x: x["RelAbundance_27"]/x["RelAbundance_17"],
                sens_fold_changes = lambda x: np.log10(x["Sensitivity_27"]/x["Sensitivity_17"]),
                sens_norm_diff = lambda x: (x["Sensitivity_27"]-x["Sensitivity_17"])/((x["Sensitivity_27"]+x["Sensitivity_17"])/2),
                abu_norm_diff = lambda x: (x["RelAbundance_27"]-x["RelAbundance_17"])/((x["RelAbundance_27"]+x["RelAbundance_17"])/2))
        )

        df_full["Category"] = df_full["Subsystem"].map(category_map)

        categories = df_full.Category.unique()

        ## Correlation of fold changes
        dicti = {
            "Accession" : [],
            "Category" : [],
            "Pcor" : [],
            "Ppval" : [],
            "P95ci_l" : [],
            "P95ci_h" : [],
            "Scor" : [],
            "Spval" : [],
            "n" : []}

        i=0
        for cat in categories:
            for acc in accessions:
                df_cur = df_full[(df_full["Category"]==cat) & (df_full["Accession"]==acc)]
                if df_cur.shape[0]<3:
                    print(f"Subsystem {cat} skipped, because of insufficient samples.")
                    continue
                pres = st.pearsonr(df_cur["abu_fold_changes"],df_cur["sens_fold_changes"])
                sres = st.spearmanr(df_cur["abu_fold_changes"],df_cur["sens_fold_changes"])
                
                dicti["Accession"].append(acc)
                dicti["Category"].append(cat)
                dicti["Pcor"].append(pres.statistic)
                dicti["Ppval"].append(pres.pvalue)
                dicti["P95ci_l"].append(pres.confidence_interval(confidence_level=0.95)[0])
                dicti["P95ci_h"].append(pres.confidence_interval(confidence_level=0.95)[1])
                dicti["Scor"].append(sres.statistic)
                dicti["Spval"].append(sres.pvalue)
                dicti["n"].append(df_cur["UniprotID"].unique().shape[0])
            i+=1

        cat_cor = pd.DataFrame(dicti)
        print(f"Subsystem specific correlations between {"significant" if sig else ""} fold changes in sensitivity and abundance")
        print(cat_cor)
        print()


    print()
    ### ==== CLUSTER SPECIFIC CORRELATION ==== 
    print("==== CLUSTER SPECIFIC CORRELATION ==== ")
    clust_order = ["Spurious",
            "High",
            "Continuous",
            "Low"]  

    for sig in [False,True]:
        if(sig):
            filt_foundation = foundation.loc[adj_pval<0.05]
        else:
            filt_foundation = foundation    
        print(f"Cluster specific correlations of fold changes {"of significantly changed protein" if sig else ""}")
        df_full = (filt_foundation
        .loc[:,["UniprotID","Accession","sens_17_mean","sens_27_mean","mean_rel_abu_17","mean_rel_abu_27","Cluster"]]
        .rename(columns =  {"sens_17_mean" : "Sensitivity_17",
                        "sens_27_mean" : "Sensitivity_27",
                        "mean_rel_abu_17" : "RelAbundance_17",
                        "mean_rel_abu_27" : "RelAbundance_27"})
        .drop_duplicates()
        .replace(0,np.nan)
        .dropna()
        .assign(abu_fold_changes = lambda x: x["RelAbundance_27"]/x["RelAbundance_17"],
                sens_fold_changes = lambda x: np.log10(x["Sensitivity_27"]/x["Sensitivity_17"]),
                sens_norm_diff = lambda x: (x["Sensitivity_27"]-x["Sensitivity_17"])/((x["Sensitivity_27"]+x["Sensitivity_17"])/2),
                abu_norm_diff = lambda x: (x["RelAbundance_27"]-x["RelAbundance_17"])/((x["RelAbundance_27"]+x["RelAbundance_17"])/2))
        )

        ## Correlation of fold changes
        dicti = {
            "Accession" : [],
            "Cluster" : [],
            "Pcor" : [],
            "Ppval" : [],
            "P95ci_l" : [],
            "P95ci_h" : [],
            "Scor" : [],
            "Spval" : [],
            "n" : []}

        i=0
        for clust in clust_order:
            for acc in accessions:
                df_cur = df_full[(df_full["Cluster"]==clust) & (df_full["Accession"]==acc)]
                if df_cur.shape[0]<3:
                    print(f"Cluster {clust} skipped, because of insufficient samples.")
                    continue
                pres = st.pearsonr(df_cur["abu_fold_changes"],df_cur["sens_fold_changes"])
                sres = st.spearmanr(df_cur["abu_fold_changes"],df_cur["sens_fold_changes"])
                
                dicti["Accession"].append(acc)
                dicti["Cluster"].append(clust)
                dicti["Pcor"].append(pres.statistic)
                dicti["Ppval"].append(pres.pvalue)
                dicti["P95ci_l"].append(pres.confidence_interval(confidence_level=0.95)[0])
                dicti["P95ci_h"].append(pres.confidence_interval(confidence_level=0.95)[1])
                dicti["Scor"].append(sres.statistic)
                dicti["Spval"].append(sres.pvalue)
                dicti["n"].append(df_cur["UniprotID"].unique().shape[0])
            i+=1

        cat_cor = pd.DataFrame(dicti)

        
        print(cat_cor)
        print()

    
    print()
    #### ==== CORRELATION OF PREDICTED AND MEASURED ABUNDANCE
    print("==== CORRELATION OF PREDICTED AND MEASURED ABUNDANCE ==== ")
    #### 1. Get predicted abundances from flux sampling data
    flux_samp_data = pd.read_csv(FLUX_SAMP_WIDE,index_col=0)
    rel_cols = ["Temperature"] + [c for c in flux_samp_data.columns if "draw_prot" in c]
    flux_samp_data = (flux_samp_data
                    .loc[flux_samp_data["Temperature"].isin([17,27]),:]
                    .loc[:,rel_cols]
                    .rename(columns=dict(zip([c for c in flux_samp_data.columns if "draw_prot" in c],[c.removeprefix("draw_prot_") for c in flux_samp_data.columns if "draw_prot" in c])))
                    .groupby("Temperature")
                    .mean()
    )
    norm_means = (flux_samp_data
                .div(flux_samp_data.sum(axis=1),axis=0)
                .transpose()
                .assign(fold_change_pabu = lambda x: np.log10(x[27] / x[17]),
                        norm_diff_pabu = lambda x: (x[27]-x[17])/((x[17]+x[27])/2))
                .rename(columns={17:"PredAbu_17",
                                27:"PredAbu_27"})
    )

    fig, axes = plt.subplots(ncols=2,figsize=(10,6),constrained_layout = True)
    #### 2. Compute fold changes / norm differences in predicted abundance relative to total protein
    test_cur = foundation.merge(norm_means.reset_index(names="UniprotID"),on="UniprotID",how="left")
    corr = {}
    axes[0].axvline(0,ls=":",color="gray")
    axes[0].axhline(1,ls=":",color="gray")

    axes[1].axvline(0,ls=":",color="gray")
    axes[1].axhline(0,ls=":",color="gray")

    alpha = 0.8
    size = 2
    for acc in accessions:
            test_fc= test_cur.loc[test_cur["Accession"]==acc,["ProteinGroup","mean_rel_abu_fold_change","fold_change_pabu"]].drop_duplicates().replace([0,np.inf,-np.inf],np.nan).dropna()
            test_nd= test_cur.loc[test_cur["Accession"]==acc,["ProteinGroup","norm_abu_diff","norm_diff_pabu"]].drop_duplicates().dropna()
            #### 3. Compare fold changes / norm differences with measurement data
            print(acc)
            print(f"Correlation between measurements and predictions")
            print(f"Fold changes (n={test_fc["ProteinGroup"].unique().shape[0]})")
            print("Pearson")
            print(st.pearsonr(test_fc["mean_rel_abu_fold_change"],test_fc["fold_change_pabu"]))
            corr[f"{acc}_fc"] = st.pearsonr(test_fc["mean_rel_abu_fold_change"],test_fc["fold_change_pabu"])
            print("Spearman")
            print(st.spearmanr(test_fc["mean_rel_abu_fold_change"],test_fc["fold_change_pabu"]))
            sns.regplot(test_fc,y="mean_rel_abu_fold_change",x="fold_change_pabu",color = palette[acc],ax=axes[0],scatter_kws={"alpha": alpha,"s":size},line_kws = {"label":acc})

            print()
            print(f"Correlation between measurements and predictions")
            print(f"Normalized differences (n={test_nd["ProteinGroup"].unique().shape[0]})")
            print("Pearson")
            print(st.pearsonr(test_nd["norm_abu_diff"],test_nd["norm_diff_pabu"]))
            corr[f"{acc}_nd"] = st.pearsonr(test_nd["norm_abu_diff"],test_nd["norm_diff_pabu"])
            print("Spearman")
            print(st.spearmanr(test_nd["norm_abu_diff"],test_nd["norm_diff_pabu"]))
            sns.regplot(test_nd,y="norm_abu_diff",x="norm_diff_pabu",color = palette[acc],ax=axes[1],scatter_kws={"alpha": alpha,"s":size})
            print()
            print()
            
            
    axes[0].set_xlabel("$log_{10}$ fold-change in predicted abundance")
    axes[0].set_ylabel("Fold-change in measured abundance")

    axes[0].annotate(xy=(0.5,0.9),text = f"$r_P$ = {np.round(corr["Bur.0_fc"].statistic,3)}, $p$ = {corr["Bur.0_fc"].pvalue:.2e}",xycoords = 'axes fraction',color = palette["Bur.0"])
    axes[0].annotate(xy=(0.5,0.8),text = f"$r_P$ = {np.round(corr["Pla.0_fc"].statistic,3)}, $p$ = {corr["Pla.0_fc"].pvalue:.2e}",xycoords = 'axes fraction',color = palette["Pla.0"])

    axes[1].set_xlabel("Norm. difference in predicted abundance")
    axes[1].set_ylabel("Norm. difference in measured abundance")
    axes[1].annotate(xy=(0.5,0.9),text = f"$r_P$ = {np.round(corr["Bur.0_nd"].statistic,3)}, $p$ = {corr["Bur.0_nd"].pvalue:.2e}",xycoords = 'axes fraction',color = palette["Bur.0"])
    axes[1].annotate(xy=(0.5,0.8),text = f"$r_P$ = {np.round(corr["Pla.0_nd"].statistic,3)}, $p$ = {corr["Pla.0_nd"].pvalue:.2e}",xycoords = 'axes fraction',color = palette["Pla.0"])

    axes[0].legend(title="Accession")

    handles,labels = axes[0].get_legend_handles_labels()
    axes[0].legend_.remove()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(0.5,0),ncols=2)
    fig.savefig(SUPP_FIG_DIR / "regression_plot_predicted_measured_abundance.png",bbox_inches="tight")


### Function definitions
def createFoundationalDataFrame(use_pert_data = False):
    ### Data prep
    ## Load proteomics
    full_prot_data = pd.read_csv(PROT_LONG,index_col=0)
    full_prot_data = full_prot_data.loc[((full_prot_data["Accession"]=="Pla.0") | (full_prot_data["Accession"]=="Bur.0")),:]

    ### How many protein groups are covered
    print(f"The proteomics data covers {full_prot_data["PG.ProteinGroups"].unique().shape[0]} protein groups.")

    ## Load model enzyme mapping
    enzyme_subsystem_map = pd.read_csv(MODEL_ENZYME_2_SUBSYSTEM,index_col=0)

    def summarize(g):
        a, b = g[27], g[17]
        return pd.Series({
            "mean_rel_abu_17": b.mean(),
            "std_rel_abu_17" : b.std(),
            "mean_rel_abu_27": a.mean(),
            "std_rel_abu_27" : a.std(),
            "fold_change": a.mean() / b.mean(),
            "difference": a.mean() - b.mean(),
            "norm_difference": (a.mean() - b.mean())/((a.mean() + b.mean())/2),
            "absolute difference": abs(a.mean() - b.mean()),
            "norm_abs_difference" : abs((a.mean() - b.mean())/((a.mean() + b.mean())/2)),
            "pval": st.ttest_ind(a, b, nan_policy="omit").pvalue
        })

    ## Load model related proteomics
    prot_data = pd.read_csv(PROTEOMICS_DATA,index_col=0)
    prot_data = prot_data.loc[((prot_data["Accession"]=="Pla.0") | (prot_data["Accession"]=="Bur.0")),:]
    proc_prot_data = (prot_data
    .loc[:,["UniprotID","ProteinGroupID","Accession","Temperature","Replicate","Relative abundance"]]
    .pivot(index = ["UniprotID","ProteinGroupID","Accession","Replicate"],columns="Temperature",values="Relative abundance")
    .reset_index()
    .groupby(["UniprotID","ProteinGroupID","Accession"])
    .apply(summarize, include_groups=False)
    .reset_index()
    )
    new_colnames = ["UniprotID","ProteinGroup","Accession","mean_rel_abu_17","std_rel_abu_17","mean_rel_abu_27","std_rel_abu_27","mean_rel_abu_fold_change","mean_rel_abu_diff","norm_abu_diff","mean_abs_rel_abu_diff","mean_norm_abs_rel_abu_diff","rel_abu_diff_pval"]
    proc_prot_data.columns = new_colnames

    na_17 = proc_prot_data["mean_rel_abu_17"].isna()
    na_27 = proc_prot_data["mean_rel_abu_27"].isna()


    print(f"{proc_prot_data.UniprotID.unique().shape[0]} model enzymes are within the processed proteomics data")

    print(f"{proc_prot_data.loc[na_17,"UniprotID"].unique().shape[0]} enzymes were not measured at 17°C")
    print(f"{proc_prot_data.loc[na_27,"UniprotID"].unique().shape[0]} enzymes were not measured at 27°C")

    proc_prot_data = proc_prot_data.loc[~na_17&~na_27,:]

    ### How many protein groups are within this subset
    print(f"The filtered proteomics data covers {proc_prot_data["ProteinGroup"].unique().shape[0]} protein groups.")
    ### How many model enzymes are within this subset
    print(f"The filtered proteomics data covers {proc_prot_data["UniprotID"].unique().shape[0]} model enzyme ids.")



    ## Load model pmet mapping
    pmet_map = pd.read_csv(MODEL_ENZYME_2_PMET,index_col=0)

    ## Load Limiting class mapping
    LC_mapping = pd.read_csv(LIMITING_CLASS_TAB,index_col=0)

    ## Load clustering results
    clustering = pd.read_csv(CLUST_RES_DIR / "enzyme_clustering.csv").loc[:,["UniProtID","Cluster"]]

    ## Load sensitivity data
    def CoV(x):
        c = (np.std(x) / np.mean(x))
        return c if not np.isnan(c) else 0

    if(use_pert_data):
        temp = [17,27]
        full_sens_data = pd.read_csv(PERT_ESC_WIDE,index_col=0)
        temp_sens_data = full_sens_data.loc[(full_sens_data["SampleID"].str.startswith(f"T{temp[0]}_") | full_sens_data["SampleID"].str.startswith(f"T{temp[1]}_")),]
        temp_sens_data["Temperature"] = temp_sens_data["SampleID"].str.replace(r"_[0-9]+","",regex=True).str.replace("T","")
        sens_data = (temp_sens_data
        .drop(columns="SampleID")
        .groupby("Temperature")
        .aggregate(["mean","std",CoV])
        .transpose()
        .reset_index(names=["Enzyme","Statistic"])
        .pivot(index = "Enzyme",columns="Statistic",values=["17","27"])
        .assign(mean_fold_change = lambda x: x[("27","mean")] / x[("17","mean")])
        )

        sens_data.columns = ["sens_" + c for c in sens_data.columns.map('_'.join).str.strip('_')]
        sens_data.reset_index(inplace=True)

    else:
        full_sens_data = pd.read_csv(ESC_DATA_WIDE,index_col=0)
        sens_data = (full_sens_data
                    .loc[:,["17","27"]]
                    .assign(sens_mean_fold_change = lambda x: x["27"] / x["17"])
                    .rename(columns = {"17":"sens_17_mean",
                                        "27":"sens_27_mean"})
                    .reset_index(names = ["Enzyme"])
                    )
    ## Create long data frame
    # ModelID | ProteinGroup | Accession | Pmet | Subsystem | rel_abu_17 | rel_abu_27 | rel_abu_fold_change | mean_sens_17 | mean_sens_27 | sens_fold_change |

    foundation = pd.merge(proc_prot_data,enzyme_subsystem_map, left_on="UniprotID",right_on="Enzyme",how="left").drop(columns="Enzyme")
    print(f"After merging with the enzyme subsystem map, the data retains {foundation["UniprotID"].unique().shape[0]} enzymes")
    foundation = foundation.merge(pmet_map, left_on = "UniprotID",right_on="Enzymes",how="left").drop(columns="Enzymes")
    print(f"After merging with the pseudometabolite map, the data retains {foundation["UniprotID"].unique().shape[0]} enzymes")
    foundation = foundation.merge(sens_data, left_on = "UniprotID",right_on="Enzyme",how="left").drop(columns="Enzyme")
    print(f"After merging with the sensitivity coefficients, the data retains {foundation["UniprotID"].unique().shape[0]} enzymes")
    foundation = foundation.merge(LC_mapping.reset_index(), on = "UniprotID",how="left")
    print(f"After merging limiting classes, the data retains {foundation["UniprotID"].unique().shape[0]} enzymes")
    foundation = foundation.merge(clustering, left_on = "UniprotID",right_on="UniProtID",how="left").drop(columns="UniProtID")
    print(f"After merging clustering, the data retains {foundation["UniprotID"].unique().shape[0]} enzymes")
    foundation.columns

    if(use_pert_data):
        foundation = foundation.loc[:,['UniprotID', 'ProteinGroup', 'Accession', 'Subsystem',
            'Name', 'Pmet','Type', 'Isorxns',
            'mean_rel_abu_17', 'std_rel_abu_17', 'mean_rel_abu_27', 'std_rel_abu_27',
            'mean_rel_abu_fold_change' ,"mean_rel_abu_diff","norm_abu_diff", "mean_abs_rel_abu_diff","mean_norm_abs_rel_abu_diff","rel_abu_diff_pval", 
            'sens_17_CoV', 'sens_17_mean', 'sens_17_std', 'sens_27_CoV', 'sens_27_mean', 'sens_27_std',
            'sens_mean_fold_change',
            "LimitingClass","Cluster"]]
    else:
        foundation = foundation.loc[:,['UniprotID', 'ProteinGroup', 'Accession', 'Subsystem',
                'Name', 'Pmet','Type', 'Isorxns', 
                'mean_rel_abu_17', 'std_rel_abu_17', 'mean_rel_abu_27', 'std_rel_abu_27',
                'mean_rel_abu_fold_change' ,"mean_rel_abu_diff","norm_abu_diff", "mean_abs_rel_abu_diff","mean_norm_abs_rel_abu_diff","rel_abu_diff_pval",
                'sens_17_mean', 'sens_27_mean', 
                'sens_mean_fold_change',
                "LimitingClass","Cluster"]]
    return [foundation, full_prot_data]


def plotIn_Outof_model_comparison(foundation, full_prot_data, palette,cax, accessions=["Bur.0","Pla.0"]):
    pgs_in_model = foundation.ProteinGroup.unique()
    pgs_not_in_model = np.setdiff1d(full_prot_data["PG.ProteinGroups"].unique(),pgs_in_model)

    test_data = (full_prot_data
    .loc[:,["PG.ProteinGroups","Accession","Temperature","Relative abundance"]]
    .groupby(["PG.ProteinGroups","Accession","Temperature"])
    .mean()
    .reset_index()
    .pivot(index=["PG.ProteinGroups","Accession"],columns="Temperature",values="Relative abundance")
    .assign(norm_diff = lambda x: (x[27]-x[17])/((x[27]+x[17])/2))
    .reset_index()
    .drop_duplicates()
    )
    test_data["In model"] = False
    test_data.loc[test_data["PG.ProteinGroups"].isin(pgs_in_model),"In model"] = True


    for acc in accessions:
        x = test_data.loc[(test_data["Accession"]==acc) & (test_data["In model"]),"norm_diff"].dropna()
        y = test_data.loc[(test_data["Accession"]==acc) & (~test_data["In model"]),"norm_diff"].dropna()


    sns.violinplot(test_data,x="In model",y = "norm_diff",hue="Accession",ax=cax,palette=palette)
    cax.axhline(0,ls=":",color ="gray")

    #print(test_data.loc[:,["PG.ProteinGroups","In model"]].drop_duplicates().value_counts("In model"))

    def stat_diff(g):
        a , b = g.loc[g["In model"],"norm_diff"] , g.loc[~g["In model"],"norm_diff"] 
        res = st.ttest_ind(a,b,nan_policy="omit",alternative="greater")
        return pd.Series({
            "statistic" : res.statistic,
            "pvalue" : res.pvalue
        })
        
    sig_df = (test_data
            .groupby("Accession")
            .apply(stat_diff,include_groups=False))


    y_offset = 0.02

    x1 = -.18
    x2 = 0.78
    y1 = 0.35
    y2 = 0.37

    cax.plot([x1, x1, x2, x2], [y1, y2, y2, y1], linewidth=1, color='k')
    cax.annotate(xy=((x1+x2)/2,y2 + y_offset),text = f"$t=${np.round(sig_df.loc["Bur.0","statistic"],2)} $p=${sig_df.loc["Bur.0","pvalue"]:.2e}",color = "k",ha="center")

    x1 = .22
    x2 = 1.22
    y1 = 0.25
    y2 = 0.27


    cax.plot([x1, x1, x2, x2], [y1, y2, y2, y1], linewidth=1, color='k')
    cax.annotate(xy=((x1+x2)/2,y2 + y_offset),text = f"$t=${np.round(sig_df.loc["Pla.0","statistic"],2)} $p=${sig_df.loc["Pla.0","pvalue"]:.2e}",color = "k",ha="center")

    cax.set_ylabel("Norm. difference in abundance between T")


    nl = [f"In ecAraCore (n={len(pgs_in_model)})" if l.get_text()=="True" else f"Not in ecAraCore (n={len(pgs_not_in_model)})" for l in cax.get_xticklabels()]
    cax.set_xticklabels(nl)

    cax.set_xlabel(None)

def plotNormDiff_clusters(foundation, ax1, ax2):
    cur_ax = [ax1,ax2]
    clust_order = ["Spurious",
                    "High",
                    "Not limiting",
                    "Continuous",
                    "Low"]

    barplot_df = (foundation
                #.loc[st.false_discovery_control(foundation["rel_abu_diff_pval"], method='bh')<0.05,:]
                .assign(norm_abu_diff = lambda x: (x["mean_rel_abu_27"]-x["mean_rel_abu_17"]) / ((x["mean_rel_abu_27"]+x["mean_rel_abu_17"])/2))
                .loc[:,["Accession","ProteinGroup","UniprotID","Cluster","norm_abu_diff"]]
                .drop_duplicates()
                
                )
    plot_bar_cluster_comp(barplot_df,split_acc=False,clust_order=clust_order,axes=cur_ax[0])

    barplot_df = (foundation
                #.loc[st.false_discovery_control(foundation["rel_abu_diff_pval"], method='bh')<0.05,:]
                .loc[:,["Accession","ProteinGroup","UniprotID","Cluster","norm_abu_diff"]]
                .drop_duplicates()
                )
    n_sig2 = plot_bar_cluster_comp(barplot_df,clust_order=clust_order,axes=cur_ax[1])

    cur_ax[0].set_ylabel("Norm. difference in rel. abundance($n_{prot}$="+f"${n_sig2}$)")
    cur_ax[1].set_ylabel(None)
    plt.setp(cur_ax[1].get_yticklabels(), visible=False)

    cur_ax[0].set_xlabel(None)
    cur_ax[1].set_xlabel(None)

def plotRegFC(foundation, palette, cax, accessions=["Bur.0","Pla.0"]):
    # normalized difference abu vs. normalized difference sens
    test_cur = (foundation
    #.loc[adj_pval<0.05]
    .loc[:,["UniprotID","Accession","mean_rel_abu_17","mean_rel_abu_27","sens_17_mean","sens_27_mean"]]
    .rename(columns =  {"sens_17_mean" : "Sensitivity_17",
                    "sens_27_mean" : "Sensitivity_27",
                    "mean_rel_abu_17" : "RelAbundance_17",
                    "mean_rel_abu_27" : "RelAbundance_27"})  
    .assign(sens_fold_change = lambda x: x["Sensitivity_27"] / x["Sensitivity_17"],
            abu_fold_change = lambda x: x["RelAbundance_27"] / x["RelAbundance_17"])
    .replace([0,np.inf],np.nan)
    .dropna()
    .drop_duplicates()
    )

    test_cur["log_sens_fold_change"]=np.log10(test_cur["sens_fold_change"])
    for acc in accessions:
        test_acc = test_cur[test_cur["Accession"]==acc]
        sns.regplot(test_acc,x="log_sens_fold_change",y="abu_fold_change",color=palette[acc],ax=cax)


    cax.axvline(0,color="gray",ls=":")
    cax.axhline(1,color="gray",ls=":")
    cax.set_xlabel('$log_{10}$(ESC fold-change)')
    cax.set_ylabel("Relative abundance fold-change")

    cor = {"Bur.0":st.pearsonr(test_cur.loc[test_cur["Accession"]=="Bur.0","log_sens_fold_change"],test_cur.loc[test_cur["Accession"]=="Bur.0","abu_fold_change"]),
        "Pla.0":st.pearsonr(test_cur.loc[test_cur["Accession"]=="Pla.0","log_sens_fold_change"],test_cur.loc[test_cur["Accession"]=="Pla.0","abu_fold_change"])}

    cax.annotate(xy=(0.5,0.9),text = f"$r_P$ = {np.round(cor["Bur.0"].statistic,3)}, $p$ = {cor["Bur.0"].pvalue:.2e}",xycoords = 'axes fraction',color = palette["Bur.0"])
    cax.annotate(xy=(0.5,0.8),text = f"$r_P$ = {np.round(cor["Pla.0"].statistic,3)}, $p$ = {cor["Pla.0"].pvalue:.2e}",xycoords = 'axes fraction',color = palette["Pla.0"])

def plotSubsystem_correlations(foundation,category_map,palette,cax,accessions=["Bur.0","Pla.0"]):
    df_full = (foundation
    .loc[:,["UniprotID","Accession","Subsystem","sens_17_mean","sens_27_mean","mean_rel_abu_17","mean_rel_abu_27"]]
    .rename(columns =  {"sens_17_mean" : "Sensitivity_17",
                    "sens_27_mean" : "Sensitivity_27",
                    "mean_rel_abu_17" : "RelAbundance_17",
                    "mean_rel_abu_27" : "RelAbundance_27"})
    .drop_duplicates()
    .replace(0,np.nan)
    .dropna()
    .assign(abu_fold_changes = lambda x: x["RelAbundance_27"]/x["RelAbundance_17"],
            sens_fold_changes = lambda x: np.log10(x["Sensitivity_27"]/x["Sensitivity_17"]),
            sens_norm_diff = lambda x: (x["Sensitivity_27"]-x["Sensitivity_17"])/((x["Sensitivity_27"]+x["Sensitivity_17"])/2),
            abu_norm_diff = lambda x: (x["RelAbundance_27"]-x["RelAbundance_17"])/((x["RelAbundance_27"]+x["RelAbundance_17"])/2))
    )

    df_full["Category"] = df_full["Subsystem"].map(category_map)

    categories = df_full.Category.unique()

    ## Correlation of fold changes
    dicti = {
        "Accession" : [],
        "Category" : [],
        "Pcor" : [],
        "Ppval" : [],
        "P95ci_l" : [],
        "P95ci_h" : [],
        "Scor" : [],
        "Spval" : [],
        "n" : []}

    for acc in accessions:
        for cat in categories:
            df_cur = df_full[(df_full["Category"]==cat) & (df_full["Accession"]==acc)]
            pres = st.pearsonr(df_cur["abu_fold_changes"],df_cur["sens_fold_changes"])
            sres = st.spearmanr(df_cur["abu_fold_changes"],df_cur["sens_fold_changes"])
            dicti["Accession"].append(acc)
            dicti["Category"].append(cat)
            dicti["Pcor"].append(pres.statistic)
            dicti["Ppval"].append(pres.pvalue)
            dicti["P95ci_l"].append(pres.confidence_interval(confidence_level=0.95)[0])
            dicti["P95ci_h"].append(pres.confidence_interval(confidence_level=0.95)[1])
            dicti["Scor"].append(sres.statistic)
            dicti["Spval"].append(sres.pvalue)
            dicti["n"].append(df_cur["UniprotID"].unique().shape[0])

    cat_cor = pd.DataFrame(dicti)
    cat_cor

    subsystem_order = cat_cor.loc[cat_cor["Accession"]=="Pla.0"].sort_values("Pcor",ascending=False).loc[:,"Category"].unique()
    cat_cor["Category"] = pd.Categorical(cat_cor["Category"],
                                        categories=subsystem_order,
                                        ordered=True)

 
    subsystem_order = cat_cor.loc[cat_cor["Accession"]=="Pla.0"].sort_values("Pcor",ascending=False).loc[:,"Category"].unique()
    cat_cor["Category"] = pd.Categorical(cat_cor["Category"],
                                        categories=subsystem_order,
                                        ordered=True)

    sns.barplot(cat_cor,y="Pcor",x="Category",hue="Accession",
                order=subsystem_order, hue_order=accessions,
                errorbar=None,ax=cax,palette=palette)

    for container, acc in zip(cax.containers, accessions):
        d = cat_cor[cat_cor["Accession"] == acc].set_index("Category").loc[subsystem_order]
        x_centers = [b.get_x() + b.get_width() / 2 for b in container]
        cax.errorbar(x_centers, d["Pcor"],
                    yerr=[d["Pcor"] - d["P95ci_l"], d["P95ci_h"] - d["Pcor"]],
                    fmt="none", ecolor="grey", elinewidth=0.5, capsize=2,
                    linestyle="--")
        # significance stars, placed relative to the error bar end
        for x, (_, row) in zip(x_centers, d.iterrows()):
            if row["Ppval"] < 0.05:
                if row["Pcor"] >= 0:
                    y, va = row["P95ci_h"], "bottom"   # above the upper whisker
                else:
                    y, va = row["P95ci_l"], "top"      # below the lower whisker
                cax.annotate("*", xy=(x, y), xytext=(0, -2 if va == "bottom" else 2),
                            textcoords="offset points", ha="center", va=va,
                            color=palette[acc], fontsize=12, fontweight="bold")

    cax.set_ylabel("Pearson correlation of fold-changes")
    cax.tick_params(axis="x",rotation=25)
    cax.axhline(0,ls=":",color="gray")
    plt.setp(cax.get_xticklabels(), ha='right')
    cax.set_xlabel("Metabolic subsystem")

def comp_cluster_differences(df):
        clust_order = ["Spurious",
                    "High",
                    "Not limiting",
                    "Continuous",
                    "Low"]
        
        dicti = {"C1":[],
                "C2":[],
                "t":[],
                "pval":[]}
        
        ## General across accessions
        # test against 0
        for i in np.arange(5):
            c1 = clust_order[i]
            x1 = df.loc[df["Cluster"]==c1,"norm_abu_diff"]
            res = st.ttest_1samp(x1,0)
            dicti["C1"].append(c1)
            dicti["C2"].append("0")
            dicti["t"].append(res.statistic)
            dicti["pval"].append(res.pvalue)

        # test cluster differences
        for i in np.arange(4):
            for j in np.arange(i+1,5):
                c1 = clust_order[i]
                c2 = clust_order[j]
                x1 = df.loc[df["Cluster"]==c1,"norm_abu_diff"]
                x2 = df.loc[df["Cluster"]==c2,"norm_abu_diff"]
                res = st.ttest_ind(x1,x2)
                dicti["C1"].append(c1)
                dicti["C2"].append(c2)
                dicti["t"].append(res.statistic)
                dicti["pval"].append(res.pvalue)

        return(pd.DataFrame(dicti))


def comp_acc_differences(df):
    clust_order = ["Spurious",
                    "High",
                    "Not limiting",
                    "Continuous",
                    "Low"]
        
    dicti = {"Cluster":[],
            "t":[],
            "pval":[]}
        
    ## Pla-0 vs. Bur-0
    for i in np.arange(5):
        c1 = clust_order[i]
        x1 = df.loc[(df["Accession"]=="Pla.0") & (df["Cluster"]==c1),"norm_abu_diff"]
        x2 = df.loc[(df["Accession"]=="Bur.0") & (df["Cluster"]==c1),"norm_abu_diff"]
        res = st.ttest_1samp(x1,0)
        dicti["Cluster"].append(c1)
        dicti["t"].append(res.statistic)
        dicti["pval"].append(res.pvalue)
    return(pd.DataFrame(dicti))

def plot_bar_cluster_comp(barplot_df,split_acc = True,clust_order = None,axes=None):    
    if clust_order is None:
        clust_order = ["Spurious",
                        "High",
                        "Continuous",
                        "Not limiting",
                        "Low"]
    barplot_df["Cluster"] = pd.Categorical(barplot_df["Cluster"],
                                        categories = clust_order,
                                            ordered=True)

    pg_per_cluster = barplot_df.loc[:,["ProteinGroup","Cluster"]].drop_duplicates().value_counts("Cluster")
    new_values = barplot_df["Cluster"].astype(str)
    new_label = []
    for cluster in clust_order:
        changed_idx = barplot_df["Cluster"]==cluster
        nl = cluster + " (n = " + pg_per_cluster[cluster].astype(str) + ")"
        new_values.loc[changed_idx] = nl
        new_label.append(nl)

    barplot_df["Cluster"] = pd.Categorical(new_values,
                                            categories = new_label,
                                            ordered=True)

    if(not split_acc):
        palette={}
        sns.barplot(barplot_df,x="Cluster",y="norm_abu_diff",
        #           hue=hue_col,
                    ax = axes, errorbar=("ci",95),
                    color = "gray",err_kws={"linewidth" : 0.5,"color":"black"}, capsize=0.1
                            )
        sns.pointplot(barplot_df, x="Cluster", y="norm_abu_diff",
                    #hue=hue_col,
                    #order=order, hue_order=accessions,
                    #dodge=0.8 - 0.8 / 2,     # total spread of the centers = bar-group width minus one bar
                    errorbar=None, linestyle=":", markers=".",color = "lightgray",
                    #palette=palette, 
                    ax=axes, legend=False)
              
    else:

        sns.barplot(barplot_df,x="Cluster",y="norm_abu_diff",hue="Accession",ax = axes,
                    errorbar=("ci",95),err_kws={"linewidth" : 0.5,"color":"gray"}, capsize=0.1
                    ,palette={"Bur.0":"#6BAED6",
                                             "Pla.0":"#FD8D3C"} 
                    )
        sns.pointplot(barplot_df, x="Cluster", y="norm_abu_diff",
                    hue="Accession",
                    #order=order, hue_order=accessions,
                    dodge=0.8 - 0.8 / 2,     # total spread of the centers = bar-group width minus one bar
                    errorbar=None, linestyle=":", markers=".",
                    palette={"Bur.0":"#C6DBEF",
                            "Pla.0":"#FDD0A2"},  
                    ax=axes, legend=False)
    '''             
    sns.stripplot(barplot_df, x="Cluster", y="norm_abu_diff", hue="Accession",
                #order=order, hue_order=accessions,
                dodge=True, alpha=0.4, jitter =0.1,
                palette= {"Bur.0":"#C6DBEF",
                            "Pla.0":"#FDD0A2"}, 
                ax=axes, legend=False)
    '''
    axes.axhline(0,color="gray",ls=":")
    axes.tick_params(axis="x",rotation=-35)
    plt.setp(axes.get_xticklabels(), ha='left')
    #axes.set_yscale("symlog", linthresh=abs(barplot_df["norm_abu_diff"].iloc[np.argmin(abs(barplot_df["norm_abu_diff"]))]))
    n = barplot_df["ProteinGroup"].unique().shape[0]
    
    return n

def comp_fc_corr(df):
        dicti = {"Limiting class":[],
                "Pearson corr":[],
                "Pearson pval":[],
                "Spearman corr":[],
                "Spearman pval":[]}
        
        for c in ["All","OLE","PLE"]:
            if c == "All":
                pres = st.pearsonr(df["log_sens_fold_change"],df["abu_fold_change"])
                sres = st.spearmanr(df["log_sens_fold_change"],df["abu_fold_change"])
            else:
                idx = df["LimitingClass"]==c
                pres = st.pearsonr(df.loc[idx,"log_sens_fold_change"],df.loc[idx,"abu_fold_change"])
                sres = st.spearmanr(df.loc[idx,"log_sens_fold_change"],df.loc[idx,"abu_fold_change"])

            dicti["Limiting class"].append(c)
            dicti["Pearson corr"].append(pres.statistic)
            dicti["Pearson pval"].append(pres.pvalue)
            dicti["Spearman corr"].append(sres.statistic)
            dicti["Spearman pval"].append(sres.pvalue)     
        return(pd.DataFrame(dicti))

if __name__ == "__main__":
    main()