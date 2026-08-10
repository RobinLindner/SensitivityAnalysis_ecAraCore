import sys
import numpy as np
import pandas as pd
import io
from pathlib import Path
root_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(root_dir))
from source.GEM import GEM
from source import MODEL_DIR, FVCB_PHI_TABLE, PERT_OUT_DIR
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


TAU = 0.001

def main():
    n_it = int(sys.argv[1])         # 1000
    max_change = float(sys.argv[2]) # 0.025
    temp = int(sys.argv[3])         # 10-40
    
    changes = dict()
    SC_dfs = list()
    flux_dfs = list()
    phi_tab = pd.read_csv(FVCB_PHI_TABLE).loc[:,["Temperature","phi"]].set_index("Temperature").squeeze()
    print(f'Starting Temperature {temp}')
    phi = phi_tab[temp]
    model = GEM(MODEL_DIR / f"TGEMAdj_{temp}.mat")
    model = addCOratioConstraint(model,phi)
    m = model.S.shape[0]
    SA = SensitivityAnalysis(model, inequality_idx= [m-2,m-1])
    SA.SetUpperEnzymebounds(1000)

    ori_kcats = SA.gem.S[np.ix_(SA.prot_idx, SA.rxn_idx)].copy()
    for i in range(n_it):
        # slightly change kcats
        curr_changes = dict()
        for pix in SA.prot_idx:
            curr_changes[pix] = 1 + max_change * ((np.random.uniform() - 0.5) * 2)
            SA.gem.S[pix, SA.rxn_idx] *= curr_changes[pix]
        changes[(temp, i)] = curr_changes

        # Sensitivity coefficients
        with silence():
            SA.ResetProblems()
            df_S = SA.ComputeSensitivityCoefficients_alt()
        df_S = df_S.loc[df_S["Type"]=="ESC",:].drop(columns = "Type")
        df_S.columns = [f'SC_T{temp}_{i}', 'ModelID']
        df_S = df_S.set_index('ModelID')
        SC_dfs.append(df_S)

        # Flux distribution
        df_Fluxes = pd.DataFrame(zip(SA.gem.rxns,SA.primal_flux_sol))
        df_Fluxes.columns = ['ModelID', f'Flux_T{temp}_{i}']
        df_Fluxes = df_Fluxes.set_index('ModelID')
        flux_dfs.append(df_Fluxes)

        # reset SA.gem.S
        SA.gem.S[np.ix_(SA.prot_idx, SA.rxn_idx)] = ori_kcats
    
    res_path = PERT_OUT_DIR / f"PerturbedESC_{temp}.tsv"
    pd.concat(SC_dfs, axis=1).to_csv(res_path, sep='\t')

    res_path = PERT_OUT_DIR / f"PerturbedFlux_{temp}.tsv"
    pd.concat(flux_dfs, axis=1).to_csv(res_path, sep='\t')

    with open(PERT_OUT_DIR / f"changes_{temp}.txt", 'w') as file:
        file.write(str(changes))


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

if __name__  == "__main__":
    main()