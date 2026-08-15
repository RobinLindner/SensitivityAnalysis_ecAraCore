import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
import scipy.stats as sp
from statsmodels.stats import multitest

import statsmodels.formula.api as smf
import statsmodels.genmod.families.family as fam
from scipy.special import logit


root_dir = Path(__file__).resolve().parents[2]

sys.path.append(str(root_dir))

from source import MODEL_DIR, PROT_RESULTS_DIR, ESC_DATA_WIDE, FLUX_SAMP_WIDE, PROTEOMICS_DATA, PERT_ESC_WIDE, MODEL_ENZYME_2_SUBSYSTEM, MODEL_ENZYME_2_PMET, SUPP_FIG_DIR, SUPP_RES_DIR
from source.GEM import GEM

## OUTPUT PATHS
CORR_ABU_PATH = PROT_RESULTS_DIR / "correlation_measured_predicted.csv"

CORR_SENS_PATH = PROT_RESULTS_DIR / "correlation_measured_sensitivity.csv"

PERTURB_CORR_PATH = PROT_RESULTS_DIR / "Spearman_correlation_perturbed_kcat.csv"

ISOENZYME_IDENTIFICATION_DF = PROT_RESULTS_DIR / "max_abu_isoenzyme_found.csv"

def main():
    flux_sampling_data = pd.read_csv(FLUX_SAMP_WIDE,index_col=0)
    model_related_proteomics = pd.read_csv(PROTEOMICS_DATA,index_col=0)

    ## Enzyme abundance means across all 100 samples
    enzyme_cols = [c for c in flux_sampling_data.columns if ("prot_" in c) & (not "prot_pool" in c)]
    prot_pool_column = [c for c in flux_sampling_data.columns if  "prot_pool" in c]

    ## 3.1.0 Draw fluxes correspond to concentrations but to compute the abundances, 
    # - we need to multiply each enzymes concentration by its molecular weight.
    model = GEM(MODEL_DIR / "TGEMAdj_20.mat")                   # load model
    MWs = [model.get_MW(e) for e in model.enzymes]              # Get MW for each enzyme
    sampled_enz_conc = flux_sampling_data.loc[:,[f"draw_prot_{e}" for e in model.enzymes]]    # Exract relevant columns
    sampled_enz_abd = sampled_enz_conc.mul(MWs,axis=1)          # Multiply concentration by molecular weight
    flux_sampling_data.loc[:,enzyme_cols] = sampled_enz_abd     # Substitute the sampled concentrations with the abundances
    rel_cols = ["SampleID","Temperature"]
    rel_cols.extend(enzyme_cols)
    abundance_data = flux_sampling_data.loc[:,rel_cols]         
    abundance_long = abundance_data.melt(id_vars = ["SampleID","Temperature"], value_name = "Predicted abundance",var_name="Enzyme")
    abundance_long["Enzyme"] = abundance_long["Enzyme"].str.removeprefix("draw_prot_")
    median_abundance_long = abundance_long.loc[:,["Temperature","Enzyme","Predicted abundance"]].groupby(["Temperature","Enzyme"]).median().reset_index()

    # Predicted abundance vectors
    abd_17 = median_abundance_long.loc[median_abundance_long["Temperature"]==17].drop(columns="Temperature").set_index("Enzyme").squeeze().rename("Predicted")
    abd_27 = median_abundance_long.loc[median_abundance_long["Temperature"]==27].drop(columns="Temperature").set_index("Enzyme").squeeze().rename("Predicted")

    # Measured abundance vectors
    cols_17 = []
    cols_27 = []
    for enzyme in model_related_proteomics.UniprotID.unique():
        enz_data = model_related_proteomics[model_related_proteomics["UniprotID"] == enzyme]
        avg_data = enz_data.loc[:,["Accession","Temperature","Relative abundance"]].groupby(["Accession","Temperature"]).mean().reset_index()
        data_17 = avg_data[avg_data["Temperature"]==17].drop(columns="Temperature").set_index("Accession").squeeze().rename(enzyme)
        data_27 = avg_data[avg_data["Temperature"]==27].drop(columns="Temperature").set_index("Accession").squeeze().rename(enzyme)
        cols_17.append(data_17)
        cols_27.append(data_27)

    m_abd_17 = pd.concat(cols_17,axis=1).transpose()
    m_abd_27 = pd.concat(cols_27,axis=1).transpose()


    # Sensitivity vectors
    SC_wide = pd.read_csv(ESC_DATA_WIDE,index_col=0)
    sens_17 = SC_wide.loc[:,"17"]
    sens_27 = SC_wide.loc[:,"27"]


    # Concatenate all vectors for correlation analysis (removes all non-overlapping enzymes)
    full_17 = m_abd_17.loc[:,["Pla.0","Bur.0"]].merge(abd_17,how="inner",left_index=True,right_index=True)
    full_17 = full_17.merge(sens_17,how="inner",left_index=True,right_index=True)
    full_17.columns = full_17.columns + "_17"
    full_27 = m_abd_27.loc[:,["Pla.0","Bur.0"]].merge(abd_27,how="inner",left_index=True,right_index=True)
    full_27 = full_27.merge(sens_27,how="inner",left_index=True,right_index=True)
    full_27.columns = full_27.columns + "_27"
    full_df = full_17.merge(full_27,how="inner",left_index=True,right_index=True)
    

    print("== SYSTEM-WIDE MEASURED vs. PREDICTED ABUNDANCE ==")
    print("Correlation between measured and predicted abundances.")
    for temp in [17,27]:
        print(f"{temp}°C:")
        for acc in ["Bur.0","Pla.0"]:
            print(f"Measured {acc} vs. predicted abundance")
            cut=full_df.loc[:,[f"{acc}_{temp}",f"Predicted_{temp}"]].dropna()
            printCorrelations(cut[f"{acc}_{temp}"],cut[f"Predicted_{temp}"])
        print()

    ## Cross-correlation btw. measured at 17° and predicted at 27° and vice versa
    print(f"Predicted abundance at 17°C vs. 27°C")
    cut = full_df.loc[:,[f"Predicted_{17}",f"Predicted_{27}"]].dropna()
    printCorrelations(cut[f"Predicted_{17}"],cut[f"Predicted_{27}"])
    print()

    print("Correlation between abundances at one temperature and predictions at the other.")
    for acc in ["Bur.0","Pla.0"]:
        for shuffle in [(17,27),(27,17)]:
            print(f"{acc} at {shuffle[0]}°C and predicted at {shuffle[1]}°C")
            cut = full_df.loc[:,[f"{acc}_{shuffle[0]}",f"Predicted_{shuffle[1]}"]].dropna()
            printCorrelations(cut[f"{acc}_{shuffle[0]}"],cut[f"Predicted_{shuffle[1]}"])
            print()

        cut = full_df.loc[:,[f"{acc}_{17}",f"{acc}_{27}"]].dropna()
        print(f"{acc} at 17°C and {acc} at 27°C")
        printCorrelations(cut[f"{acc}_{17}"],cut[f"{acc}_{27}"])
        print()


    print("== SYSTEM-WIDE MEASURED ABUNDANCE vs. SENSITIVITY ==")
    # 17°C sens vs. 27°C sens
    print("Correlation between sensitivity at 17°C and 27°C")
    printCorrelations(sens_17,sens_27)
    print()
    for temp in [17,27]:
        ## 17°C
        print(f"{temp}°C")
        # 17°C sens vs. 17°C pred
        print(f"Correlation between sensitivity and predicted abundance at {temp}°C")
        printCorrelations(full_df.loc[:,f"{temp}_{temp}"],full_df.loc[:,f"Predicted_{temp}"])
        print()
        # 17°C sens vs. 17°C measured
        print(f"Correlation between sensitivity and measured abundance at {temp}°C")
        for acc in ["Bur.0","Pla.0"]:
            print(acc)
            acc_df = full_df.loc[:,[f"{acc}_{temp}",f"{temp}_{temp}"]]
            acc_df.dropna(inplace=True)
            printCorrelations(acc_df.loc[:,f"{temp}_{temp}"],acc_df.loc[:,f"{acc}_{temp}"])
            print()

    print("== ROBUSTNESS OF SENSITIVITY CORRELATION WITH PERTURBATION DATA ==")
    print("(might take a few minutes)")

    ## - Needs to be re-done for wide data
    '''
    perturb_data = pd.read_csv(PERTURBATION_DATA_PATH,index_col=0).loc[:,["ModelID","RunID","Sensitivity Index","Temperature"]]
    perturb_data = perturb_data.loc[perturb_data["Temperature"].isin([17,27]),:]
    dicti = {"Sample":[],"Temperature":[],"Bur.0":[],"Pla.0":[]}
    for temp in [17,27]:
        temp_data = perturb_data.loc[perturb_data["Temperature"]==temp,:]
        # For sensitivity vector of sample i
        for sample in temp_data["RunID"].unique():
            sample_data = temp_data.loc[temp_data["RunID"] == sample,["ModelID","Sensitivity Index"]].set_index("ModelID").squeeze()
            dicti["Sample"].append(sample)
            dicti["Temperature"].append(temp)
            #   compute spearman correlation with measured abundance Bur.0
            dicti["Bur.0"].append(full_df[f"Bur.0_{temp}"].corr(sample_data,method="spearman"))
            #   compute spearman correlation with measured abundance Pla.0
            dicti["Pla.0"].append(full_df[f"Pla.0_{temp}"].corr(sample_data,method="spearman"))
    '''
    # - 
    perturb_data = pd.read_csv(PERT_ESC_WIDE,index_col=0)
    extracted = perturb_data["SampleID"].str.extract(r"T(\d+)_(\d+)")
    perturb_data["Temperature"] = extracted[0].astype(int)
    perturb_data["RunID"] = extracted[1].astype(int)
    perturb_data = perturb_data.loc[perturb_data["Temperature"].isin([17,27]),:]
    dicti = {"Sample":[],"Temperature":[],"Bur.0":[],"Pla.0":[]}
    for temp in [17,27]:
        temp_data = perturb_data.loc[perturb_data["Temperature"]==temp,:].drop(columns="Temperature")
        # For sensitivity vector of sample i
        for sample in temp_data["RunID"].unique():
            sample_data = temp_data.set_index("RunID").loc[sample,:]
            dicti["Sample"].append(sample)
            dicti["Temperature"].append(temp)
            #   compute spearman correlation with measured abundance Bur.0
            dicti["Bur.0"].append(full_df[f"Bur.0_{temp}"].corr(sample_data,method="spearman"))
            #   compute spearman correlation with measured abundance Pla.0
            dicti["Pla.0"].append(full_df[f"Pla.0_{temp}"].corr(sample_data,method="spearman"))
    # -
    perturb_correlation = pd.DataFrame(dicti)
    perturb_correlation.to_csv(PERTURB_CORR_PATH,index=False)

    for acc in ["Bur.0","Pla.0"]:
        for temp in [17,27]:
            data = perturb_correlation.loc[perturb_correlation["Temperature"]==temp,acc]
            print(f"{acc} at {temp}")
            print(f"Mean: {np.round(data.mean(),2)},\tMedian: {np.round(data.median(),3)},\tSD: {np.round(data.std(),3)},\tSE: {np.round(data.std()/np.sqrt(len(data)),3)},\tMin: {np.round(data.min(),3)},\tMax: {np.round(data.max(),3)}")


    print("Correlation coefficients of perturbed samples are saved at:")
    print(str(PERTURB_CORR_PATH))

    subs_map = pd.read_csv(MODEL_ENZYME_2_SUBSYSTEM,index_col=0)
    full_df_w_subs = full_df.reset_index(names="Enzyme").merge(subs_map,on="Enzyme")

    dicti= {"Subsystem":[],"Temperature":[],"Accession":[],"Type":[],"Correlation":[],"Pvalue":[],"n_e":[]}
    for sub in full_df_w_subs["Subsystem"].unique():
        sub_data = full_df_w_subs.loc[full_df_w_subs["Subsystem"]==sub,:]
        if sub_data.shape[0]<10:
            continue
        for temp in [17,27]:
            res = sp.spearmanr(sub_data[f"Predicted_{temp}"],sub_data[f"{temp}_{temp}"],nan_policy = "omit",alternative = "greater")
            dicti["Subsystem"].append(sub)
            dicti["Temperature"].append(temp)
            dicti["Accession"].append("Model")
            dicti["Type"].append("SvP")
            dicti["Correlation"].append(res.statistic)
            dicti["Pvalue"].append(res.pvalue)
            dicti["n_e"].append(len(sub_data.Enzyme.unique()))
            for acc in ["Bur.0","Pla.0"]:
                res = sp.spearmanr(sub_data[f"{acc}_{temp}"],sub_data[f"{temp}_{temp}"],nan_policy = "omit",alternative = "greater")
                dicti["Subsystem"].append(sub)
                dicti["Temperature"].append(temp)
                dicti["Accession"].append(acc)
                dicti["Type"].append("MvS")
                dicti["Correlation"].append(res.statistic)
                dicti["Pvalue"].append(res.pvalue)
                dicti["n_e"].append(len(sub_data.Enzyme.unique()))
                
                res = sp.spearmanr(sub_data[f"{acc}_{temp}"],sub_data[f"Predicted_{temp}"],nan_policy = "omit",alternative = "greater")
                dicti["Subsystem"].append(sub)
                dicti["Temperature"].append(temp)
                dicti["Accession"].append(acc)
                dicti["Type"].append("MvP")
                dicti["Correlation"].append(res.statistic)
                dicti["Pvalue"].append(res.pvalue)
                dicti["n_e"].append(len(sub_data.Enzyme.unique()))

    correlation_df = pd.DataFrame(dicti)

    # Extract wide df for sensitivity comparisons
    mean_sens_corr_df = correlation_df[correlation_df["Type"]=="MvS"].loc[:,["Accession","Subsystem","Temperature","Correlation"]]
    mean_sens_corr_wide = mean_sens_corr_df.pivot(index="Subsystem",columns=["Temperature","Accession"],values="Correlation")
    mean_sens_corr_wide.columns = [
        "_".join(map(str, col)) for col in mean_sens_corr_wide.columns
    ]
    mean_sens_corr_wide.to_csv(CORR_SENS_PATH)

    # Extract wide df for predicted abundance comparisons
    mean_pred_corr_df = correlation_df[correlation_df["Type"]=="MvP"].loc[:,["Accession","Subsystem","Temperature","Correlation"]]
    mean_pred_corr_wide = mean_pred_corr_df.pivot(index="Subsystem",columns=["Temperature","Accession"],values="Correlation")
    mean_pred_corr_wide.columns = [
    "_".join(map(str, col)) for col in mean_pred_corr_wide.columns
    ]
    mean_pred_corr_wide.to_csv(CORR_ABU_PATH)


    print("== Chi^2 TEST FOR QUALITATIVE CHANGE")
    SC_rel = SC_wide.loc[:,["17","27"]]

    significance_th = 0.05

    dicti = {
        "Accession" : [],
        "Enzyme":[], 
        "ProteinGroupID":[],
        "Avg. Measurement difference" : [],
        "MeasurementTypeOfChange":[],
        "permutation_pvalue" : [],
        "Sensitivity difference":[],
        "SensitivityTypeOfChange":[]}

    ids = model_related_proteomics.loc[:,["UniprotID","ProteinGroupID"]].drop_duplicates()
    for i in np.arange(ids.shape[0]):
        enzyme = ids.loc[:,"UniprotID"].iloc[i]
        proteinGroupID = ids.loc[:,"ProteinGroupID"].iloc[i]

        sens_diff = SC_rel.loc[enzyme,"27"] - SC_rel.loc[enzyme,"17"]
        if(abs(sens_diff)>0):
            s_type_change =  "Increase" if sens_diff > 0 else "Decrease"
        else:
            s_type_change = "Unchanged"
        for acc in ["Bur.0","Pla.0"]:
            rel_idx = (model_related_proteomics["UniprotID"] == enzyme) & (model_related_proteomics["Accession"] == acc) & (model_related_proteomics["ProteinGroupID"]==proteinGroupID)
            
            data = model_related_proteomics[rel_idx]
            
            if(len(data["Temperature"].unique())<2):
                continue

            res = sp.permutation_test(
                (data.loc[data["Temperature"]==17,"Relative abundance"], data.loc[data["Temperature"]==27,"Relative abundance"]),
                statistic=lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis),
                permutation_type='independent',
                alternative='two-sided',
                n_resamples=np.inf  # exact: tries all permutations
                )
            gb = data.loc[:,["Temperature","Relative abundance"]].groupby("Temperature").mean()
            avg_m_diff = gb.loc[27,:].values[0] - gb.loc[17,:].values[0]
            if(res.pvalue < significance_th):
                m_type_change = "Increase" if avg_m_diff > 0 else "Decrease"
            else:
                m_type_change = "Unchanged"

            dicti["Accession"].append(acc)
            dicti["Enzyme"].append(enzyme)
            dicti["ProteinGroupID"].append(proteinGroupID)
            dicti["Avg. Measurement difference"].append(avg_m_diff)
            dicti["MeasurementTypeOfChange"].append(m_type_change)
            dicti["permutation_pvalue"].append(res.pvalue)
            dicti["Sensitivity difference"].append(sens_diff)
            dicti["SensitivityTypeOfChange"].append(s_type_change)

    qual_df = pd.DataFrame(dicti)
    cont_tables =[]
    for acc in ["Bur.0","Pla.0"]:
        acc_data = qual_df[qual_df["Accession"]==acc]
        print(f"Accession: {acc}")
        print(f"\tModelenzymes measured: {len(acc_data.Enzyme.unique())}")
        print(f"\tProteinGroups measured: {len(acc_data.ProteinGroupID.unique())}")
        print(f"\t\tIncrease (significant): {sum(acc_data["Avg. Measurement difference"]>0)} ({sum(acc_data["MeasurementTypeOfChange"]=="Increase")})")
        print(f"\t\tDecrease (significant): {sum(acc_data["Avg. Measurement difference"]<0)} ({sum(acc_data["MeasurementTypeOfChange"]=="Decrease")})")
        print(f"\tSensitivity trend agreeing with measured data: {sum(acc_data["MeasurementTypeOfChange"]==acc_data["SensitivityTypeOfChange"])}")
        print(f"\tSensitivity trend not agreeing with measured data: {sum(acc_data["MeasurementTypeOfChange"]!=acc_data["SensitivityTypeOfChange"])}")
        print(pd.crosstab(acc_data.MeasurementTypeOfChange,acc_data.SensitivityTypeOfChange))
        print()
        cont_tables.append(pd.crosstab(acc_data.MeasurementTypeOfChange,acc_data.SensitivityTypeOfChange))
    i=0
    for acc in ["Bur.0","Pla.0"]:
        print(f"$\Chi^2$-Test - {acc}")
        print(sp.chi2_contingency(cont_tables[i]))
        i+=1



    print("== ISOENZYME IDENTIFICATION ==")
    isoenzyme_map = pd.read_csv(MODEL_ENZYME_2_PMET,index_col=0)
    dicti = {"Pmet":[],"nIsoMeasured":[],"Accession":[],"Temperature":[],"MaxFound":[],"MaxFoundAtOther":[],"VarianceInMeasured":[],"CoV":[]}
    for pmet in isoenzyme_map["Pmet"].unique():
        enzymes = isoenzyme_map.loc[(isoenzyme_map["Pmet"]==pmet) & (isoenzyme_map["Type"]=="Unique"),"Enzymes"].unique()
        enz_inter = np.intersect1d(enzymes,full_df.index)
        spec_df = full_df.loc[enz_inter,:]
        if(spec_df.shape[0]>1):
            for acc in ["Bur.0","Pla.0"]:
                for temp in [17,27]:
                    temp2 = 27 if temp == 17 else 17
                    idx1 = np.argmax(spec_df.loc[:,f"{acc}_{temp}"]) # index of max relative abundance
                    idx2 = np.argmax(spec_df.loc[:,f"{temp}_{temp}"]) # index of max sensitivity
                    idx3 = np.argmax(spec_df.loc[:,f"{temp2}_{temp2}"]) # index of max sensitivity at other temperature 
                    var1 = np.var(spec_df.loc[:,f"{acc}_{temp}"])
                    mean1 = np.mean(spec_df.loc[:,f"{acc}_{temp}"])
                    dicti["Pmet"].append(pmet)
                    dicti["nIsoMeasured"].append(len(enz_inter))
                    dicti["Accession"].append(acc)
                    dicti["Temperature"].append(temp)
                    dicti["MaxFound"].append(idx1==idx2)
                    dicti["MaxFoundAtOther"].append(idx1==idx3)
                    dicti["VarianceInMeasured"].append(var1)
                    dicti["CoV"].append(var1/mean1)

    maxfound_df = pd.DataFrame(dicti)
    maxfound_df.to_csv(ISOENZYME_IDENTIFICATION_DF)
    print("Data on identification of maximally abundant isoenzymes was saved:")
    print(str(ISOENZYME_IDENTIFICATION_DF))
    print()
    fig, axes = plt.subplots(ncols=2,nrows=2,constrained_layout=True)
    i=0
    for acc in ["Bur.0","Pla.0"]:
        j=0
        for temp in [17,27]:
            cut = maxfound_df.loc[(maxfound_df["Accession"]==acc) & (maxfound_df["Temperature"]==temp),:]
            max_found = cut.loc[:,["nIsoMeasured","MaxFound"]].groupby("nIsoMeasured").sum()
            length = cut.loc[:,["nIsoMeasured","MaxFound"]].groupby("nIsoMeasured").count()
            df = pd.concat([max_found,length],axis=1)
            df.columns = ["Max found","Count"]
            df["Max not found"] = df["Count"] - df["Max found"]
            df["Expected"] = df["Count"]/df.index.astype(int)
            plot_df = df.reset_index().melt(id_vars = "nIsoMeasured",value_vars = ["Max found","Max not found","Expected"],var_name="Type",value_name="n")
            sns.barplot(plot_df,x="nIsoMeasured",y="n",hue="Type",ax=axes[i,j])
            lab = ""
            axes[i,j].set_ylabel(f"{acc.replace(".","-")}:{temp}°C")
            j+=1
        i+=1

    handles, labels = axes.flat[0].get_legend_handles_labels()
    for ax in axes.flat:
        ax.legend_.remove()
        ax.set_xlabel("Isoenzymes in reaction")
    fig.legend(handles,labels,ncols=3,loc="upper center",bbox_to_anchor=(0.5,0))
    fig.savefig(SUPP_FIG_DIR / "IsoenzymesFoundBySens.png",bbox_inches = 'tight')
    

    fig, axes = plt.subplots(ncols=2,nrows=2,constrained_layout=True)
    i=0
    for acc in ["Bur.0","Pla.0"]:
        j=0
        for temp in [17,27]:
            cut = maxfound_df.loc[(maxfound_df["Accession"]==acc) & (maxfound_df["Temperature"]==temp),:]
            max_found = cut.loc[:,["nIsoMeasured","MaxFoundAtOther"]].groupby("nIsoMeasured").sum()
            length = cut.loc[:,["nIsoMeasured","MaxFoundAtOther"]].groupby("nIsoMeasured").count()
            df = pd.concat([max_found,length],axis=1)
            df.columns = ["Max found","Count"]
            df["Max not found"] = df["Count"] - df["Max found"]
            df["Expected"] = df["Count"]/df.index.astype(int)
            plot_df = df.reset_index().melt(id_vars = "nIsoMeasured",value_vars = ["Max found","Max not found","Expected"],var_name="Type",value_name="n")
            sns.barplot(plot_df,x="nIsoMeasured",y="n",hue="Type",ax=axes[i,j])
            lab = ""
            axes[i,j].set_ylabel(f"{acc.replace(".","-")}:{temp}°C")
            j+=1
        i+=1

    handles, labels = axes.flat[0].get_legend_handles_labels()
    for ax in axes.flat:
        ax.legend_.remove()
        ax.set_xlabel("Isoenzymes in reaction")
    fig.legend(handles,labels,ncols=3,loc="upper center",bbox_to_anchor=(0.5,0))
    fig.savefig(SUPP_FIG_DIR / "IsoenzymesFoundBySensAtOther.png",bbox_inches = 'tight')

    print("Isoenzyme identification performance plots were saved at:")
    print(str(SUPP_FIG_DIR / "IsoenzymesFoundBySens.png"))
    print(str(SUPP_FIG_DIR / "IsoenzymesFoundBySensAtOther.png"))
    print()

    dicti = {"Accession":[],"Temperature":[],"n_Isoenzymes":[],"n_rxns":[],"est_prop":[],"exp_prop":[],"95_ci":[],"pvalue":[]}
    for acc in ["Bur.0","Pla.0"]:
        j=0
        for temp in [17,27]:
            cut = maxfound_df.loc[(maxfound_df["Accession"]==acc) & (maxfound_df["Temperature"]==temp),:]
            max_found = cut.loc[:,["nIsoMeasured","MaxFound"]].groupby("nIsoMeasured").sum()
            lenght = cut.loc[:,["nIsoMeasured","MaxFound"]].groupby("nIsoMeasured").count()
            df = pd.concat([max_found,lenght],axis=1)
            df.columns = ["MaxFound","Count"]
            for n in df.index:
                res = sp.binomtest(df.loc[n,"MaxFound"],df.loc[n,"Count"],p=1/n,alternative="greater")
                print(f"{acc}:{temp}:{n} => pvalue: {res.pvalue}")
                dicti["Accession"].append(acc)
                dicti["Temperature"].append(temp)
                dicti["n_Isoenzymes"].append(n)
                dicti["n_rxns"].append(df.loc[n,"Count"])
                dicti["est_prop"].append(round(res.statistic,4))
                dicti["exp_prop"].append(round(1/n,4))
                dicti["95_ci"].append(round(res.proportion_ci(confidence_level=0.95).low,4))
                dicti["pvalue"].append(round(res.pvalue,4))

    binom_tests = pd.DataFrame(dicti)

    binom_tests["FDR_adj_p"] = multitest.multipletests(binom_tests["pvalue"],alpha=0.05,method="fdr_bh")[1]

    for col in ["est_prop","exp_prop","95_ci","pvalue","FDR_adj_p"]:
        binom_tests.loc[:,col] = np.round(binom_tests.loc[:,col],3)
    
    binom_tests.to_csv(SUPP_RES_DIR / "TabS1_binomial_tests.csv")
    print("Results for individual binomial tests were saved at:")
    print(str(SUPP_RES_DIR / "TabS1_binomial_tests.csv"))
    print()

    print("== GENERALIZED LINEAR MODELS ==")
    print()

    df = maxfound_df
    df["chance_logodds"] = logit(1 / df["nIsoMeasured"])   # the offset
    df["n_centered"] = df["nIsoMeasured"] - df["nIsoMeasured"].mean() #

    m1 = smf.glm(
        "MaxFound ~ 1",
        data=df,
        family = fam.Binomial(),
        offset = df["chance_logodds"]
    ).fit()
    print("=== Model 1: Using the sensitivity of the same temperature as predictors ===")
    print(m1.summary2().tables[1][["Coef.", "Std.Err.", "z", "P>|z|", "[0.025", "0.975]"]])
    beta = m1.params["Intercept"]
    ci   = m1.conf_int().loc["Intercept"]
    print(f"\nLog-odds above chance: {beta:.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"Odds ratio vs chance:  {np.exp(beta):.3f}  95% CI [{np.exp(ci[0]):.3f}, {np.exp(ci[1]):.3f}]")
    print()


    m1 = smf.glm(
        "MaxFoundAtOther ~ 1",
        data=df,
        family = fam.Binomial(),
        offset = df["chance_logodds"]
    ).fit()

    print("=== Model 2: Using the sensitivity of the other temperature as predictors ===")
    print(m1.summary2().tables[1][["Coef.", "Std.Err.", "z", "P>|z|", "[0.025", "0.975]"]])
    beta = m1.params["Intercept"]
    ci   = m1.conf_int().loc["Intercept"]
    print(f"\nLog-odds above chance: {beta:.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"Odds ratio vs chance:  {np.exp(beta):.3f}  95% CI [{np.exp(ci[0]):.3f}, {np.exp(ci[1]):.3f}]")


def printCorrelations(x,y):
    res =  sp.pearsonr(x,y,alternative="greater")
    print(f"Pearson correlation: {res.statistic}, P-value (greater):{res.pvalue}")
    res = sp.spearmanr(x,y,alternative="greater")
    print(f"Spearman correlation: {res.statistic}, P-value (greater):{res.pvalue}")

if __name__ == "__main__":
    main()