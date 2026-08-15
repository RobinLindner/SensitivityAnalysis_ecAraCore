import pandas as pd
import numpy as np
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))
from source import SUPP_RES_DIR, ESC_DATA_WIDE

COMPLEX_DATA_DIR = SUPP_RES_DIR / "complexes"

def main():
    rows=[]
    comps = ["RuBisCO","TrpS","Cytb6","bCA","ATPS","CA"]
    for comp in comps:
        sens_data = pd.read_csv(COMPLEX_DATA_DIR / f"{comp}_sensitivity.csv",index_col=0)
        row = sens_data.loc[:,["Temperature","ESC"]].groupby("Temperature").sum().transpose()
        rows.append(row)

        print(f"== {comp} == ")
        print(f"ESC at 10°C: {row.iloc[0,0]}")
        print(f"ESC at 40°C: {row.iloc[0,30]}")
        print(f"Max ESC at {row.columns[np.argmax(row.iloc[0,:])]}°C: {np.max(row.iloc[0,:])}")
        print(f"Min ESC at {row.columns[np.argmin(row.iloc[0,:])]}°C: {np.min(row.iloc[0,:])}")
        print()

        sens_wide = sens_data.pivot(columns="Temperature",index="Enzyme",values="ESC")

        print("Percent of complex sum")
        sens_wide_rel = sens_wide.div(row.loc["ESC"], axis=1)
        print(np.round(sens_wide_rel.loc[:,np.arange(10,41,2)]*100,2))
        print()

        rxn_eqs = pd.read_csv(COMPLEX_DATA_DIR / f"{comp}_rxn_assoc.csv",index_col=0)
        # reaction equation | subunits
        result = rxn_eqs.loc[:, ["Subunit", "Rxn equations"]].groupby("Rxn equations")["Subunit"].agg(
                lambda x: ", ".join(x.unique().astype(str))
            ).reset_index()
        for i in np.arange(result.shape[0]):
            print(f"Equation: {result.loc[i,"Rxn equations"]} \t\tSubunits: {result.loc[i,"Subunit"]}")

    print("== AMP deaminase ==")
    AMPD = "O80452"
    sens_data = pd.read_csv(ESC_DATA_WIDE,index_col=0)
    row = sens_data.loc[AMPD,:]
    row.index = row.index.astype(int)
    print(f"ESC at 10°C: {row.loc[10]}")
    print(f"ESC at 40°C: {row.loc[40]}")
    print(f"Max ESC at {row.index[np.argmax(row)]}°C: {np.max(row)}")
    print(f"Min ESC at {row.index[np.argmin(row)]}°C: {np.min(row)}")
    print()
    print(row.loc[np.arange(10,41,2)])
    rows.append(pd.DataFrame(row).transpose())


    print("== SUMMARY ==")

    vis_tab = pd.concat(rows,axis=0)
    comps.append("AMPD")
    vis_tab.index = comps
    print(np.round(vis_tab.loc[:,np.arange(10,41,2)],4))







if __name__ == "__main__":
    main()