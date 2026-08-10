import sys, os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))
from source import MODEL_DIR, FIG_DIR, PROT_RESULTS_DIR, PROTEOMICS_DATA, PROT_LONG, ESC_DATA_WIDE, FLUX_SAMP_WIDE, MODEL_ENZYME_2_PMET, MODEL_ISO_COMPLEX_MAP
from source.GEM import GEM
import copy
import matplotlib.gridspec as gridspec
import itertools

## Output path for the figure
FIG_OUTPATH = FIG_DIR / "proteomics_sensitivity_est_abundance.png"


rbcl_subunits = ["O03042",
                        "P10795",
                        "P10796",
                        "P10797",
                        "P10798"]

cyt_b6_subunits = ["Q9ZR03",
                        "P56775",
                        "P56776",
                        "O48717",
                        "P61039",
                        "P56774",
                        "P56771",
                        "P56773"]
beta_CA_subunits = ["P27140",
                    "P42737",
                    "Q9ZUC2",
                    "Q94CE4",
                    "Q94CE3",
                    "Q9C6F5"]
ca_subunits = ['F4JIK2', 'Q8L817', 'O04846', 'Q94CE4', 'Q9C6F5', 'P27140', 'Q94CE3',
                    'F4HUC4', 'Q9FYE3', 'Q9SUB4', 'F4IHR4', 'Q9ZUC2', 'P42737']
TrpS_subunits = ["O22765",
                "Q42529",
                "P14671",
                "P25269"]

ATPS = ['P09468', # all chloroplastic
        'P19366',
        'P56757',
        'P56758',
        'P56759',
        'P56760',
        'Q01908',
        'Q9SSS9',
        'Q01909']

AMP_deaminase = ["O80452"]

atpH = ["P56760"]


subunits = {"RuBisCo" : rbcl_subunits,
            "Cytochrome b6f" : cyt_b6_subunits,
            "Beta carbonic anhydrase" : beta_CA_subunits,
            "Carbonic anhydrase" : ca_subunits,
            "Tryptophan synthase" : TrpS_subunits,
            "AMP deaminase" : AMP_deaminase,
            "ATP synthase (subunit c)" : atpH,
            "ATP synthase": ATPS}


def main():
   
    model_related_proteomics = pd.read_csv(PROTEOMICS_DATA,index_col=0)
    unique_Ids = model_related_proteomics.loc[:,["UniprotID","ProteinGroupID"]].drop_duplicates()
      
    ## Combine the protein group Ids for both UniProt and Protein description matches for further processing 
    # 1. Get the Group Ids from the Uniprot matching (i.e. direct gene association)
    unique_Ids = model_related_proteomics["UniprotID"].unique()
    unique_Ids = model_related_proteomics.loc[:,["UniprotID","ProteinGroupID"]].drop_duplicates()
    dfs = list()
    for key, group in subunits.items():
        print(f"{sum([1 for id in group if id in unique_Ids["UniprotID"].to_numpy()])}/{len(group)} Uniprot Ids of {key} \t from the model in the proteomics data")
        matches = unique_Ids.set_index("UniprotID").loc[[id for id in group if id in unique_Ids["UniprotID"].to_numpy()],"ProteinGroupID"]
        df = pd.DataFrame({
            "ProteinGroupID": matches,
            "Complex": key,
            "Source": "Uniprot Matching"
        }).reset_index(names="UniprotID")
        dfs.append(df)
    temp = pd.concat(dfs)


    ## 0. For each accession
    accessions = ["Bur.0","Pla.0"]
    ## 1. Axes 1: Observed abundance
    ## 1.1 Get the proteomics data for the identified consolodated protein groups for 17° and 27°
    proteomics_long = pd.read_csv(PROT_LONG,index_col=0)
    proteomics_pg = proteomics_long.loc[proteomics_long["PG.ProteinGroups"].isin(temp["ProteinGroupID"]),:]
    proteomics_pg_acc = proteomics_pg.loc[proteomics_pg["Accession"].isin(accessions),:]

    ## 1.2 Plot the reaction norm for each enzyme complex, averaging over all proteins within.
    plot_data_prot = proteomics_pg_acc.merge(temp, 
                                        how = "inner", 
                                        left_on="PG.ProteinGroups", 
                                        right_on="ProteinGroupID")

    plot_data_prot["Temperature"] = plot_data_prot["Temperature"].astype(str)
    plot_data_prot["Complex"] = plot_data_prot["Complex"].replace("RuBisCo","RuBisCO small subunits")
    plot_data_prot["Complex"] = plot_data_prot["Complex"].replace("Cytochrome b6f","Cytochrome b6-f complex")


    #cut_subunits = {complex : [unit for unit in plot_data_prot.loc[plot_data_prot["Complex"]==complex,"UniprotID"].unique() if str(unit) !='nan'] for complex in plot_data_prot["Complex"].unique()}
    cut_subunits = copy.deepcopy(subunits)
    cut_subunits["RuBisCo"].remove("O03042")


    

    ## 2. Axes 2: Estimated sensitivity
    ESC_wide_10_40_summed = pd.read_csv(ESC_DATA_WIDE, index_col=0)

    ## sensitivity should be
    # 1. summed within the complex, and then
    rows = {}
    for complex, units in cut_subunits.items():
        complex_subset = ESC_wide_10_40_summed.loc[units,:]
        summed_ESC = complex_subset.sum(axis=0)
        norm_summed_ESC = summed_ESC.sub(summed_ESC.min())
        norm_summed_ESC = norm_summed_ESC.div(summed_ESC.max()-summed_ESC.min())
        rows[complex] = norm_summed_ESC
    norm_complex_ESC = pd.DataFrame(rows).T
    

    plot_data_sens = (norm_complex_ESC
                    .loc[:,np.arange(17,28).astype(str)]
                    .reset_index(names="Complex")
                    .melt(id_vars="Complex",
                        value_name="Sensitivity",
                        var_name="Temperature")
    )

    plot_data_sens["Complex"] = plot_data_sens["Complex"].replace("RuBisCo","RuBisCO small subunits")
    plot_data_sens["Complex"] = plot_data_sens["Complex"].replace("Cytochrome b6f","Cytochrome b6-f complex")

    plot_data_sens["Temperature"] = plot_data_sens["Temperature"].astype(int)


    ## 3. Axes 3: Estimated abundance
    ## 3.1 Get the abundance data for UniprotIDs associated with the complexes for all temperatures between 17° and 27°
    flux_sampling_data = pd.read_csv(FLUX_SAMP_WIDE)
    flux_sampling_data.drop(columns=flux_sampling_data.columns[0],inplace=True)

    ## Enzyme abundance means across all 100 samples
    enzyme_cols = [c for c in flux_sampling_data.columns if ("prot_" in c) & (not "prot_pool" in c)]
    prot_pool_column = [c for c in flux_sampling_data.columns if  "prot_pool" in c]

    ## 3.1.0 Draw fluxes correspond to concentrations but to compute the abundances, 
    # - we need to multiply each enzymes concentration by its molecular weight.
    model = GEM(MODEL_DIR / "TGEMAdj_20.mat")
    MWs = [model.get_MW(e) for e in model.enzymes]
    enzyme_cols = [f"draw_prot_{e}" for e in model.enzymes]
    sampled_enz_conc = flux_sampling_data.loc[:,enzyme_cols]
    sampled_enz_abd = sampled_enz_conc.mul(MWs,axis=1)


    # - then we can sum the isoenzmyes
    ## 3.1.1 Identify the relevant pseudometabolites to add the abundances of isoenzymes together for a more continuous norm.
    # Summing isoenzymes -> estimating sample statistics
    isoenzyme_pmet_map_long = pd.read_csv(MODEL_ISO_COMPLEX_MAP,index_col=0)
    flat_rel_uniprotID = list(itertools.chain.from_iterable(subunits.values()))

    
    rel_ids_mix = pd.DataFrame({"UniprotID" : flat_rel_uniprotID}).drop_duplicates().merge(isoenzyme_pmet_map_long,
                                                                            left_on = "UniprotID",
                                                                            right_on = "Isoenzymes",
                                                                            how="left")

    query_id = rel_ids_mix["ComplexID"]
    missing_idx = query_id.isna()
    query_id[missing_idx] = rel_ids_mix.loc[:,"UniprotID"] 

    rel_ids_mix["query_id"] = query_id

    cols = {}
    complexes = {}
    for complex, units in cut_subunits.items():
        consolidation_ids = rel_ids_mix.loc[:,["UniprotID","query_id"]].set_index("UniprotID").loc[units,:]
        for id in consolidation_ids["query_id"].unique():
            enzymes = consolidation_ids.index[consolidation_ids["query_id"] == id]
            enzyme_cols = [f"draw_prot_{e}" for e in enzymes]
            cols[id] = sampled_enz_abd.loc[:,enzyme_cols].sum(axis=1)
            complexes[id] = complex

    abd_df = pd.DataFrame(cols)

    # - Then we express the abundance as the proportion of the total (normalizing it).
    prot_pool_draw = (flux_sampling_data.
                      loc[:,[c for c in flux_sampling_data.columns if  "prot_pool" in c]].
                      to_numpy())                               # Vector of sampled protein draw values
    fs_rel_abd = abd_df.div(prot_pool_draw,axis=0)              # Abundance matrix scaled by total protein draw          

    sum_rel_abd_long=pd.concat([flux_sampling_data.iloc[:,0:2],
                            fs_rel_abd],
                            axis=1).melt(id_vars=["SampleID","Temperature"],
                                        value_name="Estimated abundance",
                                        var_name="Enzyme")

    sum_rel_abd_long["Complex"] = sum_rel_abd_long["Enzyme"].map(complexes)
    sum_rel_abd_long["Complex"] = sum_rel_abd_long["Complex"].replace("RuBisCo","RuBisCO small subunits")
    sum_rel_abd_long["Complex"] = sum_rel_abd_long["Complex"].replace("Cytochrome b6f","Cytochrome b6-f complex")
    


    # Mean
    #mean = sum_rel_abd_long.drop(columns="SampleID").groupby(["Temperature","Enzyme"]).mean()
    # Standard error: confidence in mean
    #sem = sum_rel_abd_long.drop(columns="SampleID").groupby(["Temperature","Enzyme"]).sem()
    # Standard deviation: actual spread of data
    #std = sum_rel_abd_long.drop(columns="SampleID").groupby(["Temperature","Enzyme"]).std()

    #est_enz_abundance_long = pd.concat([mean,sem,std],axis=1).reset_index()
    #est_enz_abundance_long.columns = ["Temperature","Enzyme","Mean abundance","Sem abundance","Std abundance"]
    #est_enz_abundance_temp = est_enz_abundance_long.loc[est_enz_abundance_long["Temperature"].isin(np.arange(17,28)),:]

    


    plot_data_est_abd = sum_rel_abd_long[sum_rel_abd_long["Temperature"].isin(np.arange(17,28))]
    ##### Enzyme | Complex | Temperature | Estimated abundance [%PC]


    ## 3.2 Plot the reaction norm for each enzyme complex across the 17° - 27° range, averaging over all proteins within.

    ### 4. Plot everything in a single figure.

    palette_first = ["#54a87f",
                "#c75a93",
                "#75ab3d",
                "#8275cb",
                "#b68f40",
                "#cc5a43"]
    palette_add = ["#cb6a49",
                "#a46cb7",
                "#7aa457"]

    palette = {"RuBisCO small subunits": palette_first[0],
            "Tryptophan synthase":palette_first[1],
            "Carbonic anhydrase" : palette_first[2],
            "Cytochrome b6-f complex" : palette_first[3],
            "AMP deaminase" : palette_first[4],
            "ATP synthase" : palette_first[5]
            }

    hue_order = palette.keys()

    fig = plt.figure(figsize=(8,5),constrained_layout=True)
    gs  = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 3])  # left:right = 1:3

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1], sharex=ax3)

    ## Proteomics
    ax_acc = [ax1,ax2]
    i=0
    for acc in accessions:
        acc_data = plot_data_prot[plot_data_prot["Accession"]==acc] 
        
        #atpH needs to be plotted individually since there is no direct uniprot match through genes
        '''
        acc_data_atpH =  acc_data[acc_data["Complex"]=="ATP synthase (subunit c)"]
        sns.lineplot(acc_data_atpH,
                    x="Temperature",
                    y="Relative abundance",
                    hue="Complex",
                    hue_order=hue_order,
                    palette=palette,
                    errorbar=None,
                    linestyle="dashed",
                    legend=False,
                    alpha=0.6,
                    ax=ax_acc[i])
        '''
        #acc_data2 = acc_data[np.invert(acc_data["Source_x"].isna())]
        sns.lineplot(acc_data,
                    x="Temperature",
                    y="Relative abundance",
                    hue="Complex",
                    hue_order=hue_order,
                    palette=palette,
                    errorbar="se",
                    ax=ax_acc[i])

        if(i==0):
            handles, labels = ax_acc[i].get_legend_handles_labels()
        
        ax_acc[i].get_legend().remove()
        ax_acc[i].grid(True)
        ax_acc[i].set_ylabel(f"Relative abundance - {acc.replace(".","-")}")
        i+=1

    ## Sensitivity 
    sns.lineplot(plot_data_sens,
                x="Temperature",
                y="Sensitivity",
                hue="Complex",
                hue_order=hue_order,
                palette=palette,
                alpha=0.9,
                ax=ax3)

    ## Estimated abundance
    sns.lineplot(plot_data_est_abd,
                x="Temperature",
                y="Estimated abundance",
                hue="Complex",
                hue_order=hue_order,
                palette=palette,
                alpha=0.9,
                errorbar=("ci",95),
                ax=ax4)
    ax4.set_yscale("log")

    for axes in [ax3,ax4]:
        axes.grid(True)
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        axes.legend_.remove()
        axes.set_xticks([17,19,21,23,25,27])

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax1.xaxis.get_label(), visible=False)
    plt.setp(ax3.xaxis.get_label(), visible=False)

    ax3.set_ylabel("Normalized sensitivity")
    ax4.set_ylabel("Predicted abundance [%PC]")
    #ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    #ax2.set_xticklabels([f"{lab}°C" for lab in ax2.get_xticklabels()])
    #ax4.set_xticklabels([f"{lab}°C" for lab in ax4.get_xticklabels()])

    ax2.set_xlabel("Temperature [°C]")
    ax4.set_xlabel("Temperature [°C]")

    fig.legend(handles,labels,loc = "upper center",bbox_to_anchor=(0.5,0),ncols=3)
    fig.text(0,1,"A",fontdict={"weight":"bold","size":14})
    fig.text(0.3,1,"B",fontdict={"weight":"bold","size":14})
    fig.text(0.3,0.52,"C",fontdict={"weight":"bold","size":14})
    fig.get_layout_engine().set(w_pad=0.1, wspace=0.1)
    fig.savefig(FIG_OUTPATH,dpi=300,bbox_inches = 'tight')

    plot_data_prot.to_csv(PROT_RESULTS_DIR / "F3_complex_proteomics_data.csv")
    plot_data_sens.to_csv(PROT_RESULTS_DIR / "F3_complex_sens_data.csv")
    plot_data_est_abd.to_csv(PROT_RESULTS_DIR / "F3_complex_abund_data.csv")
    
if __name__=="__main__":
    main()