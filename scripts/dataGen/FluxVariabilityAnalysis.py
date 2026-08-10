import numpy as np
import pandas as pd
import sys
from pathlib import Path
import cobra.io as cio
import cobra

root_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(root_dir))
from source import MODEL_DIR, FVCB_PHI_TABLE, FVA_DATA_LONG

TAU = 0.0001

def main():
    fva_frames = []
    for temp in np.arange(10,41):
        phi_tab = pd.read_csv(FVCB_PHI_TABLE).loc[:,["Temperature","phi"]].set_index("Temperature").squeeze()
        phi = phi_tab[temp]
    
        model = cio.load_matlab_model(MODEL_DIR / f"TGEMAdj_{temp}.mat")
        expr_le = - model.reactions.arm_RBO_h + model.reactions.arm_RBC_h * phi * (1-TAU)
        constraint_le = model.problem.Constraint(expr_le, ub=0)
        model.add_cons_vars(constraint_le)
        expr_le = + model.reactions.arm_RBO_h - model.reactions.arm_RBC_h * phi * (1+TAU)
        constraint_le = model.problem.Constraint(expr_le, ub=0)
        model.add_cons_vars(constraint_le)
        
        cobra_fva=cobra.flux_analysis.flux_variability_analysis(model,fraction_of_optimum=0.95)
        cobra_fva.columns = ["Min flux","Max flux"]
        cobra_fva["Temperature"]=temp
        cobra_fva.reset_index(names="Reaction")
        fva_frames.append(cobra_fva)

    full_cobra_fva = pd.concat(fva_frames)

    full_cobra_fva.to_csv(FVA_DATA_LONG)


if __name__=="__main__":
    main()