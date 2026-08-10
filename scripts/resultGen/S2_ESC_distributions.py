import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path


root_dir = Path(__file__).resolve().parents[2]

sys.path.append(str(root_dir))

from source import SUPP_FIG_DIR, ESC_DATA_WIDE

def main():
    EnzymeSensitivityAcrossTemp = pd.read_csv(ESC_DATA_WIDE,index_col=0)

    fig, ax = plt.subplots(ncols=3,sharey=True,constrained_layout = True,figsize=(10,3))
    temps = [10,25,40]
    barcolor = "#8cc5e3"
    mincolor = "#4a2377"
    mediancolor = "#0d7d87"
    maxcolor = "#f55f74"
    textsize=12
    for i in np.arange(3):
        sc_at_t = EnzymeSensitivityAcrossTemp[str(temps[i])]
        sc_at_t = sc_at_t[sc_at_t>0]

        sns.histplot(x=sc_at_t,log_scale=True,ax=ax.flat[i],bins = 20,kde=True, color = barcolor)

        ax.flat[i].vlines(np.median(sc_at_t),0,40,color = mediancolor,linestyle="dotted")
        ax.flat[i].text(np.median(sc_at_t),40,f"{np.median(sc_at_t):.1e}",color = mediancolor,ha="right", size=textsize)  
        
        ax.flat[i].vlines(np.min(sc_at_t),0,40,color = mincolor,linestyle="dotted")
        ax.flat[i].text(np.min(sc_at_t),40,f"{np.min(sc_at_t):.1e}",color = mincolor,ha="left", size=textsize)  
        
        ax.flat[i].vlines(np.max(sc_at_t),0,40,color = maxcolor,linestyle="dotted")
        ax.flat[i].text(np.max(sc_at_t),40,f"{np.max(sc_at_t):.1e}",color = maxcolor,ha="right", size=textsize)  
        
        ax.flat[i].set_xlim(1e-9,1)
        ax.flat[i].set_ylim(0,45)
        ax.flat[i].set_xlabel(f"Sensitivity at {temps[i]}°C",size=textsize)
        ax.flat[i].set_ylabel("Count", size=textsize)

    fig.savefig(SUPP_FIG_DIR / "ESC_distribution_at_select_temperatures.png")

    dicti = {"Temperature":[],
            "Min":[],
            "q10" : [],
            "q25" : [],
            "Median":[],
            "q75" : [],
            "q90" : [],
            "Max":[]}
    for temp in np.arange(10,41):
        sc_at_t = EnzymeSensitivityAcrossTemp[str(temp)]
        sc_at_t = sc_at_t[sc_at_t>0]
        dicti["Temperature"].append(temp)
        dicti["Min"].append(np.min(sc_at_t))
        dicti["q10"].append(np.quantile(sc_at_t,0.1))
        dicti["q25"].append(np.quantile(sc_at_t,0.25))
        dicti["Max"].append(np.max(sc_at_t))
        dicti["q75"].append(np.quantile(sc_at_t,0.75))
        dicti["q90"].append(np.quantile(sc_at_t,0.90))
        dicti["Median"].append(np.median(sc_at_t))
        
    pop_stats = pd.DataFrame(dicti)

    violincolor = "#8cc5e3"
    mincolor = "#4a2377"
    mediancolor = "#0d7d87"
    maxcolor = "#f55f74"

    line_plot_df = pop_stats.melt(id_vars="Temperature",var_name="Statistic",value_name="Sensitivity")
    line_plot_df["Temperature"] = line_plot_df["Temperature"].astype(int)

    fig, axes = plt.subplots()
    sns.lineplot(line_plot_df,
                x="Temperature",
                y="Sensitivity",
                hue="Statistic",
                ax=axes)
    '''palette={"Min":mincolor,
            "Median":mediancolor,
            "Max":maxcolor}'''
                        #)


    select_temps = [10,15,20,25,30,35,40]
    violin_plot_df = EnzymeSensitivityAcrossTemp.loc[:,[str(t) for t in select_temps]].reset_index(names="Enzyme").melt(id_vars="Enzyme",var_name = "Temperature",value_name="Sensitivity")
    violin_plot_df = violin_plot_df.loc[violin_plot_df["Sensitivity"]>0]
    violin_plot_df["Temperature"] = violin_plot_df["Temperature"].astype(int)
    sns.violinplot(violin_plot_df,
                x="Temperature",
                y="Sensitivity",
                ax=axes,
                log_scale=(False,True),
                native_scale=True,
                color=violincolor)

    axes.set_yscale("log")
    axes.grid(True)


    fig.savefig(SUPP_FIG_DIR / "ESC_distribution_across_temperature_range.png")


if __name__ == "__main__":
    main()