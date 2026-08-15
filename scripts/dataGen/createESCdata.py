import numpy as np
import pandas as pd
import sys, io
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))
from source import MODEL_DIR, ESC_DATA_WIDE, FVCB_PHI_TABLE
from source.GEM import GEM
from source.SensitivityAnalysis import SensitivityAnalysis

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
    columns = []
    for temp in np.arange(10,41):
        phi = phi_tab[temp]
        model = GEM(MODEL_DIR / f"TGEMAdj_{temp}.mat")
        model = addCOratioConstraint(model,phi)
        m = model.S.shape[0]
    
        with silence():
            SA = SensitivityAnalysis(model,inequality_idx= [m-2,m-1])
            SA.SolvePrimal()
            ESCs = SA.ComputeSensitivityCoefficients_alt()
            ESCs = ESCs.loc[ESCs["Type"]=="ESC", ["SensitivityCoefficient","ModelID"]].set_index("ModelID").squeeze()
            ESCs.index = ESCs.index.str.removeprefix("draw_prot_")
        columns.append(ESCs)

    ESC_data = pd.concat(columns,axis=1)
    ESC_data.columns = np.arange(10,41)
    ESC_data.to_csv(ESC_DATA_WIDE)

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