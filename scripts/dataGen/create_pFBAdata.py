import numpy as np
import pandas as pd
from pathlib import Path
import sys,io

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))

from source import MODEL_DIR, PFBA_DATA, FVCB_PHI_TABLE
from source.SensitivityAnalysis import SensitivityAnalysis
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

TAU = 0.0001

def main():
    phi_tab = pd.read_csv(FVCB_PHI_TABLE).loc[:,["Temperature","phi"]].set_index("Temperature").squeeze()
    flux_vecs = []
    for temp in np.arange(10,41):
        print(f"Running pFBA at {temp}°C")
        phi = phi_tab[temp]
        model = GEM(MODEL_DIR / f"TGEMAdj_{temp}.mat")
        model = addCOratioConstraint(model,phi)
        m = model.S.shape[0]        
        SA = SensitivityAnalysis(model,inequality_idx= [m-2,m-1])
        with silence():
            flux_vecs.append(SA.solvePFBA(alpha=0.95))
    full_df = pd.concat(flux_vecs,axis=1)
    print(full_df.head())
    full_df.columns = np.arange(10,41,1)
    pFBA_long = full_df.reset_index(names="Variable").melt(id_vars="Variable",var_name="Temperature",value_name="Flux")
    pFBA_long.to_csv(PFBA_DATA)

def addCOratioConstraint(model,phi):
    new_row = np.zeros(model.S.shape[1])
    new_row[model.get_rxn_by_id("arm_RBO_h")] = -1
    new_row[model.get_rxn_by_id("arm_RBC_h")] = phi * (1-TAU)
    model.S = np.vstack([model.S,
                        new_row])

    new_row = np.zeros(model.S.shape[1])
    new_row[model.get_rxn_by_id("arm_RBO_h")] = 1
    new_row[model.get_rxn_by_id("arm_RBC_h")] = -1*(phi * (1+TAU))
    model.S = np.vstack([model.S,
                        new_row])

    model.b = np.zeros(model.S.shape[0])
    model.mets = np.concat([model.mets,["phi_lower","phi_upper"]])
    return(model)

if __name__ == "__main__":
    main()