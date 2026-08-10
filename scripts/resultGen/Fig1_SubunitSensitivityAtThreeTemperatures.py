import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, io
from pathlib import Path
import warnings
import matplotlib.gridspec as gridspec

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))
from source import FIG_DIR, UNIPROT_ID_2_DATA, ESC_DATA_WIDE, MODEL_ID_NAME_MAP

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


# Path to save the figure
out_path = FIG_DIR / "ESC_for_complexes_three_temperatures.png"

# Patterns that we search protein names for to assign complex annotation
patterns = {"Ribulose bisphosphate carboxylase" : ["Ribulose bisphosphate carboxylase", "Rbcl", "rbcl"],
            "Tryptophan synthase": ["Tryptophan synthase"],
            "Beta carbonic anhydrase": ["Beta carbonic anhydrase"],
            "Cytochrome b6-f complex": ["Cytochrome b6-f","Cyt b6-f","Cytochrome b6f","Cytochrome b6","Cytochrome f"],
            "AMP deaminase":["AMP deaminase"]
            }

def main():
    enzyme_names = pd.read_csv(UNIPROT_ID_2_DATA, sep="\t")
    SC_wide = pd.read_csv(ESC_DATA_WIDE,index_col=0)

    enz_dfs = []
    for complex,pats in patterns.items():
        pattern = "|".join(pats)
        indices = np.where(enzyme_names["Protein names"].str.contains(pattern))[0]
        enzymes = enzyme_names.iloc[indices,:]
        with warnings.catch_warnings(action="ignore"):
            enzymes.loc[:,"Complex"] = complex
        print(f"UniprotIDs matched to the {complex} complex:")
        print(enzymes.loc[:,["Entry","Protein names"]])
        print()
        enz_dfs.append(enzymes)

    ## ATP synthase (chloroplast)
    AtpS_subunits = ['P09468', 'P19366', 'P56757', 'P56758', 'P56759', 'P56760', 'Q01908', 'Q9SSS9','Q01909']
    atps_enzymes = enzyme_names[enzyme_names["Entry"].isin(AtpS_subunits)]
    with warnings.catch_warnings(action="ignore"):
        atps_enzymes.loc[:,"Complex"] = "ATP synthase"
    print(f"UniprotIDs matched to the ATPS complex:")
    print(atps_enzymes.loc[:,["Entry","Protein names"]])
    print()
    enz_dfs.append(atps_enzymes)

    # Join Enzyme complex data frames 
    enzymes = pd.concat(enz_dfs,axis=0)

    enzymes.drop(["Gene Names","Entry Name"],axis=1,inplace=True)
    enzyme_names["ProteinName_cut"] = enzyme_names["Protein names"].str.replace(r"\s\(.*$","",regex=True)

    # Get SC of interest from temperature specific SC data
    SC_eoi= SC_wide.loc[enzymes["Entry"],["10","20","40"]].reset_index(names="Entry")

    # Get col sums of remaining SC data
    SC_remaining = SC_wide.drop(index=enzymes["Entry"]).loc[:,["10","20","40"]]
    SC_sums = SC_remaining.sum(axis=0)

    # Summarize remaining into "Other" complex
    other = pd.DataFrame([{"Entry": "Other",
                        "Protein names" : "Other",
                        "ProteinName_cut": "Other",
                        "Complex": "Other",
                        "10": SC_sums.iloc[0],
                        "20": SC_sums.iloc[1],
                        "40": SC_sums.iloc[2]}])

    wide_df = pd.concat([pd.merge(enzymes,SC_eoi,on="Entry"),other],axis=0)

    wide_df_comp = wide_df.loc[:,["Complex","10","20","40"]].groupby("Complex").sum().reset_index()

    wide_df["Compartment"] = "Cytosolic/Unspecified"
    wide_df.loc[wide_df["Protein names"].str.contains(", chloroplastic"),"Compartment"] = "Chloroplastic"
    wide_df.loc[wide_df["Protein names"].str.contains(", mitochondrial"),"Compartment"] = "Mitochondrial"

    plot_df = wide_df.melt(id_vars=["Entry","Complex"],value_vars=["10","20","40"],value_name="ESC",var_name="Temperature")

    plot_df_comp = wide_df_comp.melt(id_vars=["Complex"],value_vars=["10","20","40"],value_name="ESC",var_name="Temperature")

    complex_dict = dict(zip(wide_df["Entry"],wide_df["Complex"]))

    palette_first = ["#54a87f",
                "#c75a93",
                "#75ab3d",
                "#8275cb",
                "#b68f40",
                "#cc5a43"]
    palette_add = ["#cb6a49",
                "#a46cb7",
                "#7aa457"]

    palette = {"Ribulose bisphosphate carboxylase": palette_first[0],
            "Tryptophan synthase":palette_first[1],
            "Beta carbonic anhydrase" : palette_first[2],
            "Cytochrome b6-f complex" : palette_first[3],
            "AMP deaminase" : palette_first[4],
            "ATP synthase" : palette_first[5],
            "Other" : "gray",
            "Chloroplastic" : palette_add[2],
            "Mitochondrial": palette_add[0],
            "Cytosolic/Unspecified" : palette_add[1]
            }

    #pd.Series(nl).to_csv(DATA_DIR / "AraCore_supporting_data/Uniprot_to_plottingLabel.csv")

    id_map = pd.read_csv(MODEL_ID_NAME_MAP)
    id_map = id_map[id_map["Type"]=="EnzymeDraw"]
    id_map["ModelID"] = id_map["ModelID"].str.removeprefix("draw_prot_")
    genes = id_map.set_index("ModelID").loc[:,"Name"].squeeze()
    genes["O48717"] = "petM"
    plot_df["second_label_name"] = plot_df["Entry"].map(genes)

    nl = dict(zip(plot_df["Entry"],plot_df["second_label_name"]))

    nl["Other"] = "-"

    # group all atpS subunits that are not atpH
    atps_df = plot_df.loc[(plot_df["Complex"]=="ATP synthase") & (plot_df["Entry"]!="P56760"),["Temperature","ESC"]].groupby("Temperature").sum().reset_index()
    plot_df.drop(index = plot_df.index[(plot_df["Complex"]=="ATP synthase") & (plot_df["Entry"]!="P56760")],inplace=True)
    atps_df["Entry"] = "Other ATPS"
    atps_df["Complex"] = "ATP synthase"
    atps_df["second_label_name"] = "Other ATPS (n=8)"
    plot_df = pd.concat([plot_df,atps_df.loc[:,plot_df.columns]])

    complex_dict["Other ATPS"] = "ATP synthase"
    nl["Other ATPS"] = "-"

    plot_df = plot_df.groupby("Complex", sort=False).apply(lambda g: g.sort_values("second_label_name"),include_groups=False)

    # ==== CELL 2 ====
    fig = plt.figure(figsize=(10,8))

    outer_gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.1,height_ratios=[1,2])

    upper_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[0], wspace=0.1)

    axes_top = [fig.add_subplot(gs) for gs in upper_gs]

    for ax in axes_top:
        ax.tick_params(axis="x",labelbottom=False)

    for i in [1,2]:
        #axes_top[i].sharex(axes_top[0])
        axes_top[i].sharey(axes_top[0])
        axes_top[i].tick_params(labelleft=False)
        
    order=[k for k in palette.keys()]
    #order[order.index("ATP synthase (subunit c)")] = "ATP synthase"
    #palette["ATP synthase"] = palette["ATP synthase (subunit c)"]
    i=0
    for temp in [10,20,40]:
        axes_top[i].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        axes_top[i].set_axisbelow(True)  
        subplot_df = plot_df_comp[plot_df_comp["Temperature"]==str(temp)]
        subplot_df.loc[:,"Complex"] = subplot_df["Complex"].replace("ATP synthase (subunit c)","ATP synthase")
        
        sns.barplot(data=subplot_df,
                    y="Complex",
                    x="ESC",
                    orient="h",
                    hue="Complex",
                    palette=palette,
                    ax=axes_top[i],
                    legend=False,
                    order=order[:7])
        axes_top[i].set_xlabel(f"ESC at {temp}°C")
        axes_top[i].set_ylabel("Enzyme")
        axes_top[i].set_xticks([0,0.2,0.4])
        axes_top[i].set_xlim((0,0.6))
        #ax[i].set_xscale("log")
        i+=1

    lower_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[1], wspace=0.1)
    axes_bottom = [fig.add_subplot(gs) for gs in lower_gs]

    #for ax in axes_bottom:
    #    ax.sharex(axes_top[0])
    axes_bottom[0].sharex(axes_top[0])
    for i in [1,2]:
        axes_bottom[i].sharex(axes_top[i])
        axes_bottom[i].sharey(axes_bottom[0])
        axes_bottom[i].tick_params(labelleft=False)


    i=0
    for temp in [10,20,40]:
        axes_bottom[i].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        axes_bottom[i].set_axisbelow(True)  
        subplot_df = plot_df[plot_df["Temperature"]==str(temp)]
        sns.barplot(data=subplot_df,y="Entry",x="ESC",orient="h",hue="Complex",palette=palette,ax=axes_bottom[i])
        axes_bottom[i].set_xlabel(f"ESC at {temp}°C")
        axes_bottom[i].set_ylabel("UniprotID")
        #ax[i].set_xscale("log")
        i+=1

    #handles, labels = ax[0].get_legend_handles_labels()

    axes = axes_top + axes_bottom
    # Remove individual legends
    for ax in axes:
        if(ax.legend_ is not None):
            ax.legend_.remove()

    # first set of y axis labels
    y_axis_labels = []
    for label in axes_bottom[0].get_yticklabels():
        label.set_bbox(dict(
            facecolor=palette[complex_dict[label.get_text()]],
            edgecolor="none",
            alpha=0.3,
            boxstyle="round,pad=0.06"
        ))
        y_axis_labels.append(label.get_text())

    ax2 = axes_bottom[0].twinx()
    ax2.set_ylim(axes_bottom[0].get_ylim())          # match the original y axis range
    ax2.set_yticks(axes_bottom[0].get_yticks())      # match tick positions
    ax2.set_yticklabels([nl[slab] for slab in y_axis_labels])      # set your custom labels
    ax2.set_ylabel("Subunit")
    ax2.yaxis.set_tick_params(length=0)  # hide tick marks if desired
    ax2.yaxis.set_label_position("left")
    ax2.yaxis.tick_left()

    # Offset outward from the existing left spine
    ax2.spines["left"].set_position(("outward", 120))  # increase 80 for more distance
    ax2.spines["right"].set_visible(False)

    fig.text(-0.2,0.86,s="A",fontdict={"weight":"bold","size":14})
    fig.text(-0.2,0.59,s="B",fontdict={"weight":"bold","size":14})
    fig.savefig(out_path,dpi=300,bbox_inches="tight")

if __name__ == "__main__":
    main()