import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, io
from pathlib import Path
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import warnings
import textwrap


root_dir = Path(__file__).resolve().parents[2]

sys.path.append(str(root_dir))

from source import RESULT_DIR, MODEL_DIR, SUPP_FIG_DIR, SUPP_RES_DIR, ESC_DATA_WIDE, FLUX_SAMP_WIDE, MODEL_ID_NAME_MAP, GENE_SHORT_NAMES, UNIPROT_ID_2_NAME
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


# supp data out path
sdata_out_path = SUPP_RES_DIR / "complexes"

COMPLEX_TAB_PATH = RESULT_DIR / "tables/SensitivityAnalysis/complex_subunit_table.csv"

FIG_DIR = SUPP_FIG_DIR / "ComplexSensitivity"

# Baseline model file
model_path = MODEL_DIR / "TGEMAdj_20.mat"


# Switches for replotting
switches = {"RuBisCO":True,
            "TrpS":True,
            "Cytb6":True,
            "bCA":True,
            "CA":True,
            "ATPS":True}

log_scale_eta = {"RuBisCO":False,
            "TrpS":True,
            "Cytb6":False,
            "bCA":False,
            "CA":False,
            "ATPS":False}

log_scale_abu = {"RuBisCO":False,
            "TrpS":True,
            "Cytb6":False,
            "bCA":False,
            "CA":False,
            "ATPS":False}

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
            "CA": ['F4JIK2', 'Q8L817', 'O04846', 'Q94CE4', 'Q9C6F5', 'P27140', 'Q94CE3',
                   'F4HUC4', 'Q9FYE3', 'Q9SUB4', 'F4IHR4', 'Q9ZUC2', 'P42737'],
            "ATPS":['P09468', # c
                    'P19366', # c
                    'P56757', # c
                    'P56758', # c
                    'P56759', # c
                    'P56760', # c
                    'Q01908', # c
                    'Q9SSS9', # c
                    'Q01909'] # c
                        }

complex_names = {"RuBisCO":"RuBisCO",
                 "TrpS": "Tryptophan synthase",
                 "Cytb6":"Cytochrome b6-f",
                 "bCA":"Beta carbonic anhydrases",
                 "CA":"Carbonic anhydrases",
                 "ATPS": "ATP synthase"}

def main():

    model_name_map = pd.read_csv(MODEL_ID_NAME_MAP,index_col=0).loc[:,"Name"].squeeze().to_dict()
    db_name_map = pd.read_csv(UNIPROT_ID_2_NAME,index_col=0).squeeze().to_dict()
    dicti = {"Complex":[],
            "Subunit":[],
            "Short name":[],
            "Long name":[]}
    for key, entry in subunits.items():
        for sub in entry:
            dicti["Complex"].append(complex_names[key])
            dicti["Subunit"].append(sub)
            dicti["Short name"].append(model_name_map[sub])
            dicti["Long name"].append(db_name_map[sub])

    pd.DataFrame(dicti).to_csv(COMPLEX_TAB_PATH)

    for comp, enzymes in subunits.items():
        if switches[comp]:
            print(f"Processing data for {comp}")
            fig, rxnflux_frame, abu_frame, sens_frame, eff_frame, rxn_eq_frame = plot_subunit_properties(enzymes,
                                                                                                         log_scale_abu[comp],
                                                                                                         log_scale_eta[comp])
            fig.savefig(FIG_DIR / f"{comp}_subunit_comparison.png",dpi=300,bbox_inches = "tight")
            saveSuppData(sdata_out_path / f"{comp}", rxnflux_frame,abu_frame, sens_frame, eff_frame, rxn_eq_frame)

    return

def saveSuppData(outpath, rxnflux_frame, abu_data, sens_data, eff_data, rxn_eq_data):
    rxnflux_frame.to_csv(str(outpath)+"_reaction_fluxes.csv")
    abu_data.to_csv(str(outpath)+"_predicted_abundance.csv")
    sens_data.to_csv(str(outpath)+"_sensitivity.csv")
    eff_data.to_csv(str(outpath)+"_efficiency.csv")
    rxn_eq_data.to_csv(str(outpath)+"_rxn_assoc.csv")
    return

def plot_subunit_properties(subunits,log_abu,log_eta):
    '''
    Given a list of enzymes in "subunits" plots a 4 panel figure showing temperature dependence of
        1. Flux through reactions catalyzed by the enzymes  - flux sampling data
        2. Predicted abundance of the enzymes               - flux sampling data
        3. Sensitivity coefficients of the enzymes          - Sensitivity data
        4. Efficiency eta of enzymes                        - Temperature adjusted models 
    
    Returns:
        1. Figure object
        2. Formatted enzyme abundance data frame
        3. Formatted enzyme sensitivity data frame
        4. Formatted enzyme efficiency data frame
        5. Enzyme reaction association data frame 
    '''

    flux_sampling_df = pd.read_csv(FLUX_SAMP_WIDE)
    ESC_wide = pd.read_csv(ESC_DATA_WIDE,index_col=0)
    model = GEM(str(model_path))

    AraCore_id_map = pd.read_csv(MODEL_ID_NAME_MAP,index_col=0)
    enz_id_map = AraCore_id_map[AraCore_id_map["Type"]=="EnzymeDraw"]
    enz_id_map.index = enz_id_map.index.str.removeprefix("draw_prot_")
    genes = enz_id_map.loc[:,"Name"].squeeze()
    genes["O48717"] = "petM"

    labels = genes.loc[subunits].to_dict()

    ## Get Kcat and enzyme efficiency
    dfs = list()
    for temp in np.arange(10,41):
        model_t = GEM(MODEL_DIR / f"TGEMAdj_{temp}.mat")
        kcat = [model_t.get_single_kcat(sub) for sub in subunits]
        MWs = [model_t.get_MW(sub) for sub in subunits]
        eff = np.array(kcat) / np.array(MWs)
        kcat_df = pd.DataFrame({"Subunit":subunits,
                                "Kcat":kcat,
                                "Molecular weight":MWs,
                                "Efficiency":eff})
        kcat_df["Temp"] = temp
        dfs.append(kcat_df)
    kcat_df = pd.concat(dfs,axis=0)

    
    ## Get reaction equation frame
    sub_dfs = list()
    for sub in subunits:
        rxns = model.get_catalyzed_rxns(sub)
        rxn_equations = np.array([model.get_rxn_equation(rxn) for rxn in rxns]).reshape(-1,)
        sub_df = pd.DataFrame({
            "Subunit": np.repeat(sub,len(rxns)),
            "Reactions" : rxns,
            "Rxn equations": rxn_equations
        })
        sub_dfs.append(sub_df)

    rxn_eq_df =pd.concat(sub_dfs,axis=0)
    
    

    ## Get potential arm reactions for subunits.
    # 1. Get the pseudometabolites from the rxn equations
    rxn_eq_df_s = rxn_eq_df.loc[:,["Reactions","Rxn equations"]].drop_duplicates()
    pmets = []
    reactions = []
    for i in np.arange(rxn_eq_df_s.shape[0]):
        rxn_eq = rxn_eq_df_s["Rxn equations"].iloc[i]
        if("pmet" in rxn_eq):
            pmet, = [met for met in rxn_eq.split(" ") if "pmet" in met]
            pmets.append(pmet)
        else:
            reactions.append(rxn_eq_df_s["Reactions"].iloc[i])


    # 2. Get the arm reactions from these
    
    for pmet in pmets:
        reactions.extend([model.rxns[i] for i in model.get_met_rxns(pmet) if "arm" in model.rxns[i]])
    reactions = list(set(reactions))
    reactionNames = AraCore_id_map.loc[reactions,"Name"]
    

    ## Get the sampled reaction flux for arm reactions
    enzymes = ["draw_prot_" + s for s in subunits]
    plot_rxnflux = flux_sampling_df.loc[:,["SampleID","Temperature"]+reactions].melt(id_vars = ["SampleID","Temperature"],value_vars=reactions,var_name="Reaction",value_name="Flux")
    for i in np.arange(len(reactions)):
        plot_rxnflux.loc[plot_rxnflux["Reaction"] == reactions[i],"Reaction"] = reactions[i]+ "-" +reactionNames.iloc[i]

    ##Get the sampled enzyme concentrations  
    plot_abundance = (flux_sampling_df.
        loc[:,np.concatenate([["SampleID","Temperature"], enzymes])].
        melt(id_vars = ["SampleID","Temperature"],
             value_vars=enzymes,
             var_name="Enzyme",
             value_name="Abundance"))
    
    plot_abundance.loc[:,"Enzyme"] = plot_abundance["Enzyme"].str.removeprefix("draw_prot_")

    ## Get the sensitivity data
    ESC_long = (ESC_wide.
                reset_index(names="Enzyme").
                melt(id_vars="Enzyme",
                     value_vars=ESC_wide.columns,
                     value_name="ESC",
                     var_name="Temperature"))

    plot_esc = ESC_long[ESC_long["Enzyme"].isin(subunits)]
    plot_esc.loc[:,"Temperature"] = plot_esc["Temperature"].astype(int)

    # Get the enzyme efficiency data
    plot_efficiency = kcat_df[kcat_df["Temp"].isin(np.arange(10,41))]
    plot_efficiency.loc[:,"Enzyme"] = [labels[e] if e in labels.keys() else e for e in plot_efficiency["Subunit"] ]


    # Define a palette for an arbitrary number of enzymes
    palette_keys1 = plot_rxnflux["Reaction"].unique().tolist()
    palette_keys2 = np.sort(plot_efficiency["Enzyme"].unique()).tolist()

    all_keys = palette_keys1 + palette_keys2  # combined, no overlap assumed
    #print(set(palette_keys1) & set(palette_keys2))
    with warnings.catch_warnings(action="ignore"):
        cmap = cm.get_cmap("tab20", len(all_keys))
    master_palette = {label: cmap(i) for i, label in enumerate(all_keys)}


    plot_abundance.loc[:,"Enzyme"] = plot_abundance["Enzyme"].map(plot_efficiency.set_index("Subunit").loc[:,"Enzyme"].drop_duplicates())
    plot_esc.loc[:,"Enzyme"] = plot_esc["Enzyme"].map(plot_efficiency.set_index("Subunit").loc[:,"Enzyme"].drop_duplicates())

    ## Plot everything
    fig, ax = plt.subplots(ncols=2,nrows=2,constrained_layout=True,sharex=True,figsize=(9,6))

    sns.lineplot(plot_rxnflux,
                 x="Temperature",
                 y="Flux",
                 hue = "Reaction",
                 ax=ax[0,0],
                 palette=master_palette,
                 legend=False,
                 style="Reaction",
                 dashes = [(2, 1), (1, 1)],
                 alpha = 0.5)

    sns.lineplot(plot_abundance,x="Temperature",y="Abundance",hue = "Enzyme",ax=ax[0,1],palette=master_palette,legend=False,alpha=0.8)

    sns.lineplot(plot_esc,x="Temperature",y="ESC",hue="Enzyme",ax=ax[1,0],palette=master_palette,legend=False,alpha=0.8)

    sns.lineplot(plot_efficiency,x="Temp",y="Efficiency",hue="Enzyme",ax=ax[1,1],palette=master_palette,legend=False,alpha=0.8)    

    ax[0,0].set_ylabel("Metabolic flux [$\\frac{mmol}{gDW} \\cdot \\frac{1}{h}$]")

    ax[0,1].set_ylabel("Enzyme concentration [$\\frac{mmol}{gDW}$]")
    ax[1,0].set_xlabel("Temperature [°C]")

    ax[1,1].set_ylabel("Enzyme efficiency $\\eta$")
    ax[1,1].set_xlabel("Temperature [°C]")

    if(log_abu):
        ax[0,1].set_yscale("log")

    if(log_eta):
        ax[1,1].set_yscale("log")
    
    for a in ax.flat:
        a.grid(True)
        if a.get_legend() is not None:
            a.get_legend().remove()
    
    all_keys = list(dict.fromkeys(palette_keys1 + palette_keys2))

    handles = [Line2D([0],[0], color=master_palette[k], label=k) for k in all_keys]
    labels = ["rxn: " + k if k in palette_keys1 else "enz: " + k for k in all_keys]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1))

    gene_name = pd.read_csv(GENE_SHORT_NAMES,index_col=0).squeeze().to_dict()
    gpr_rules = []
    text=""
    for rxn in reactions:
        gr_rule = get_gpr_for_rxn(rxn,model,gene_name)
        gpr_rules.append(f"{rxn}: " + gr_rule)
        text = text + "\n" + rxn + ":\n" + textwrap.fill(gr_rule, width=40) + "\n"
        
        #print(f"Reaction: {rxn} : {gr_rule}")

    #wrapped = textwrap.fill("\n\n".join(gpr_rules), width=40)
    fig.text(1.01,0,text,fontsize=10, va = "bottom",
         bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"))

    return fig, plot_rxnflux, plot_abundance, plot_efficiency, plot_esc , rxn_eq_df
    
def get_gpr_for_rxn(rxn_id,model,geneID_map):
    rule = model.get_grRule(rxn_id)[0]
    #print(rule)
    for pattern, repl in geneID_map.items():
        rule = rule.replace(pattern, repl)
    return rule


if __name__ == "__main__":
    main()