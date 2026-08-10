import numpy as np
import pandas as pd
import sys, io, re
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))
from source import MODEL_DIR, ESC_DATA_WIDE, MODEL_ENZYME_2_PMET
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



# Baseline model file
model_path = MODEL_DIR / "TGEMAdj.mat"

def main():

    SC_wide = pd.read_csv(ESC_DATA_WIDE,index_col=0)

    model = GEM(model_path)

    enz_pmet_map = pd.read_csv(MODEL_ENZYME_2_PMET,index_col=0)

    rxns = [rxn for rxn in model.rxns if not ("draw_prot" in rxn or "prot_pool" in rxn or "arm_" in rxn)]
    arm_rxns = [rxn for rxn in model.rxns if "arm_" in rxn]
    im_ex_rxns = [r for r in rxns if (("Im_" in r) | ("Ex_" in r))]

    cat_rxns = []
    for e in model.enzymes:
        cat_rxns.extend(model.get_catalyzed_rxns(e))

    cat_rxns = list(set(cat_rxns))

    pmets = pd.Series([m for m in model.mets if "pmet" in m])
    enz = pd.Series([m for m in model.mets if "prot" in m])
    mets = [m for m in model.mets if not "prot" in m and not "pmet" in m]

    n_spont = len(rxns)-len(im_ex_rxns)

    print("== MODEL SUMMARY ==")

    print("The ecAraCore model contains:")
    print(f"\t{len(model.mets)} constraints")
    print(f"\t{len(model.rxns)} variables")
    print("\t6 compartments")
    print(f"\t{len(np.unique([re.sub("_[hclpm]","",m) for m in mets]))} metabolites")
    print(f"\t{len(rxns)} reactions")
    print(f"\t{len(model.enzymes)} enzymes")
    print(f"\t{len(model.genes)} genes")
    print(f"\t{len(im_ex_rxns)} / {len(rxns)} reactions are import or exchange reactions.")
    print(f"\t{len(cat_rxns)} / {n_spont} ({np.round(len(cat_rxns)/n_spont*100,3)}%) reactions in the model are catalyzed by enzymes.")
    print(f"\t{len(np.unique(enz_pmet_map.loc[:,"Isorxns"]))} / {n_spont} ({np.round(len(np.unique(enz_pmet_map.loc[:,"Isorxns"])) / n_spont * 100,3)}) reactions are isoreactions contained within;")
    print(f"\t{len(arm_rxns)} arm reactions / complex reaction systems.")
    

    print("\n")

    #print([re.sub("_[hclpm]","",m) for m in pmets])
    #print(len(np.unique([re.sub("_[hclpm]","",m) for m in pmets])))

    print("== SENSITIVITY AT 20°C ==")
    sensitivity_20 = SC_wide.loc[:,"20"]
    print("At 20°C")
    print(f"\tNon-zero ESC:\t{sum(sensitivity_20!=0)} ({np.round(sum(sensitivity_20!=0)/len(sensitivity_20)*100,3)}%)")
    print(f"\tZero ESC:\t{sum(sensitivity_20==0)}  ({np.round(sum(sensitivity_20==0)/len(sensitivity_20)*100,3)}%)")

    nz_sens = sensitivity_20[sensitivity_20!=0]
    print(f"\tMedian: {np.median(nz_sens)},\tMin:{np.min(nz_sens)},\tMax:{np.max(nz_sens)}")
    print()
    
    print(f"Enzymes with an ESC of at least 0.01 at 20°C: {sum(sensitivity_20>0.01)}/671.")


if __name__ == "__main__":
    main()
