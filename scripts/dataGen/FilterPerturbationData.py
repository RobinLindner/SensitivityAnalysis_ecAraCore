import sys , re, ast
import pandas as pd
import numpy as np
from pathlib import Path

root_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(root_dir))
from source import RESULT_DIR , DATA_DIR, PERT_OUT_DIR, PERT_ESC_WIDE, PERT_FLUX_WIDE, PERT_CHANGES_WIDE



#FIL_PERT_DATA = DATA_DIR / "sEnz_datasets/Perturbation_data_filtered.tsv"

#PERT_DATA_DIR = RESULT_DIR / "tables/PerturbationAnalysis/"

#UNF_PERT_DATA = DATA_DIR / "sEnz_datasets/PtotConstr_TemperatureSensitivityAnalysis.varyingkcats_long.tsv"

#P_ESC_DATA_WIDE = DATA_DIR / "sEnz_datasets/Perturbation_ESC_data_wide.csv"

#P_FLUX_DATA_WIDE = DATA_DIR / "sEnz_datasets/Perturbation_Flux_data_wide.csv"

#P_CHANG_DATA_WIDE = DATA_DIR / "sEnz_datasets/Perturbation_Changes_wide.csv"

def main():
    ESC_dfs = []
    Flux_dfs = []
    changes_dfs = []
    for temp in np.arange(10,41):
        print(temp)
        ESC_file = PERT_OUT_DIR / f"PerturbedESC_{temp}.tsv"
        Flux_file = PERT_OUT_DIR / f"PerturbedFlux_{temp}.tsv"
        changes_file = PERT_OUT_DIR / f"changes_{temp}.txt"

        ESC_df = pd.read_csv(ESC_file,sep="\t",index_col=0)
        ESC_df.index = ESC_df.index.str.removeprefix("draw_prot_")
        ESC_df = ESC_df.transpose()
        ESC_df.index = ESC_df.index.str.removeprefix("SC_")
        ESC_df = ESC_df.reset_index(names="SampleID")

        ESC_dfs.append(ESC_df)

        
        Flux_df = pd.read_csv(Flux_file,sep="\t",index_col=0)
        Flux_df = Flux_df.transpose()
        Flux_df.index = Flux_df.index.str.removeprefix("Flux_")
        Flux_df = Flux_df.reset_index(names="SampleID")

        Flux_dfs.append(Flux_df)

        with open(changes_file, "r") as f:
            content = f.read()

        changes_df = pd.DataFrame(parse_numpy_repr_dict(content))
        changes_df.columns = np.arange(0,1000)
        changes_df = changes_df.sort_index(ascending=True).transpose()
        changes_df.index = temp + changes_df.index
        changes_dfs.append(changes_df.reset_index(names="SampleID"))

    full_ESC_frame = pd.concat(ESC_dfs)
    full_Flux_frame = pd.concat(Flux_dfs)
    full_changes_frame = pd.concat(changes_dfs)

    invalid_sample_idxs = np.where(full_ESC_frame.drop(columns="SampleID").sum(axis=1) > 1)[0]
    print(invalid_sample_idxs)
    print(f"{len(invalid_sample_idxs)} / 31000 ({np.round((len(invalid_sample_idxs) / full_ESC_frame.shape[0]) * 100,2)}%) of samples have an ESC sum larger than 1 and were removed.")

    filt_ESC_frame = full_ESC_frame.drop(full_ESC_frame.index[invalid_sample_idxs])
    filt_Flux_frame = full_Flux_frame.drop(full_Flux_frame.index[invalid_sample_idxs])
    filt_changes_frame = full_changes_frame.drop(full_changes_frame.index[invalid_sample_idxs])

    filt_ESC_frame.to_csv(PERT_ESC_WIDE)
    filt_Flux_frame.to_csv(PERT_FLUX_WIDE)
    filt_changes_frame.to_csv(PERT_CHANGES_WIDE)


def parse_numpy_repr_dict(content):
    """Same as read_numpy_repr_dict, but takes the raw string directly."""
    # Matches np.int64(123), np.float64(1.23), np.float32(...), np.int32(...),
    # np.uint8(...), etc. -- covers the common numpy scalar dtypes.
    # Captures the inner numeric literal (handles optional sign,
    # decimals, and scientific notation like 1.2e-05).
    pattern = re.compile(
        r"np\.(?:int|uint|float)\d*\(\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*\)"
    )
    cleaned = pattern.sub(r"\1", content)
 
    # Also handle bare numpy NaN/inf reprs if present (np.float64(nan) would
    # already be stripped to "nan" by the pattern above only if it matched
    # digits -- nan/inf need separate handling since they aren't \d).
    cleaned = re.sub(r"np\.(?:float\d*)\(\s*nan\s*\)", "float('nan')", cleaned)
    cleaned = re.sub(r"np\.(?:float\d*)\(\s*-?inf\s*\)", lambda m: "float('-inf')" if "-" in m.group(0) else "float('inf')", cleaned)
 
    return ast.literal_eval(cleaned)

    

if __name__ == "__main__":
    main()