import sys, os
import pandas as pd
import numpy as np
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))
from source import RESULT_DIR, FLUX_SAMP_DIR, FVA_DATA_LONG


OUTPATH = RESULT_DIR / "tables/FluxSampling/coverage_of_flux_sampling.csv"

def main():

    table_df = []
    fva_data = pd.read_csv(FVA_DATA_LONG,index_col=0)
    fva_data.reset_index(names="Reaction",inplace=True)
    fva_data.rename({"Min flux":"Min Flux",
                    "Max flux":"Max Flux"},
                    axis=1,
                    inplace=True)
    print("FVA data prepared.")

    sample_sizes = [100,1000,10000]
    for n in sample_sizes:
        sample_dfs = list()
        files = os.listdir(FLUX_SAMP_DIR / f"n{n}")
        for file in files:
            if file.startswith("."):
                continue
            temp = file.replace(f"fluxes_nsamples.{n}_vbio.0.95_","").replace(".csv","")
            temp_samples = pd.read_csv(str(FLUX_SAMP_DIR)+f"/n{n}/"+file,index_col=0)
            info_cols = pd.DataFrame({"SampleID": temp_samples.index})
            info_cols["Temperature"] = int(temp)
            temp_samples = pd.concat([info_cols,temp_samples],axis=1)
            sample_dfs.append(temp_samples)

        flux_sampling_df = pd.concat(sample_dfs,axis=0).sort_values(["Temperature","SampleID"])

        dicti = {"Temperature":[],"Coverage":[]}
        for temp in np.arange(10,41):
            dicti["Temperature"].append(temp)
            dicti["Coverage"].append(coverage(temp,fva_data,flux_sampling_df))
        coverage_df = pd.DataFrame(dicti).set_index("Temperature")
        print(f"Sample size: {n}")
        print(f"Average coverage across temperatures: {np.round(np.mean(coverage_df["Coverage"]),4)}({np.round(np.median(coverage_df["Coverage"]),4)})±{np.round(np.std(coverage_df["Coverage"]),4)}")
        table_df.append(coverage_df)
    table=pd.concat(table_df,axis=1)
    table.columns = ["$n=100$","$n=1000$","$n=10000$"]
    table.to_csv(OUTPATH)


def gap(rxn_id,flux_samples,temp):
    samples = flux_samples.loc[flux_samples["Temperature"]==temp,rxn_id]
    samples.sort_values(inplace=True)
    gaps = samples.iloc[1:].to_numpy() - samples.iloc[:-1].to_numpy()
    return(np.max(gaps))

def coverage(temp,fva_data,flux_samples):
    temp_data = fva_data.loc[fva_data["Temperature"]==temp,:]
    nz_fluxes = temp_data.loc[(temp_data["Min Flux"]!=0) | (temp_data["Max Flux"]!=0),"Reaction"]
    nz_fluxes = [rxn_id for rxn_id in nz_fluxes if "draw_prot" not in rxn_id]
    scaled_gaps = [gap(rxn,flux_samples,temp) / (temp_data.loc[temp_data["Reaction"]==rxn,"Max Flux"].item() - temp_data.loc[temp_data["Reaction"]==rxn,"Min Flux"].item())  for rxn in nz_fluxes]
    return 1 - (1/len(nz_fluxes)) * sum(scaled_gaps)

if __name__ == "__main__":
    main()
