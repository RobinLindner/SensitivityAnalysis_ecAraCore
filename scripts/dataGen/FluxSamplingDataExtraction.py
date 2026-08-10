import sys, os
import pandas as pd
from pathlib import Path
root_dir = Path(__file__).resolve().parents[3]

sys.path.append(str(root_dir))
from source import FLUX_SAMP_DIR

n = 10000

def main():
    for n in [100,1000,10000]:
        sample_dir = FLUX_SAMP_DIR / f"n{n}/"
        sample_dfs = list()
        files = os.listdir(sample_dir)
        for file in files:
            if file.startswith("."):
                continue
            temp = file.replace(f"fluxes_nsamples.{n}_vbio.0.95_","").replace(".csv","")
            temp_samples = pd.read_csv(str(sample_dir)+"/"+file,index_col=0)
            info_cols = pd.DataFrame({"SampleID": temp_samples.index + "_" + temp})
            temp_samples = pd.concat([info_cols,temp_samples],axis=1)
            sample_dfs.append(temp_samples)
        flux_sampling_df = pd.concat(sample_dfs,axis=0).sort_values(["Temperature","SampleID"])
        flux_sampling_df.to_csv(FLUX_SAMP_DIR / f"/flux_sampling_n.{n}_a.95_T.10_40.csv")

if __name__ == "__main__":
    main()
