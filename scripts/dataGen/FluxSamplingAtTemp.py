import sys
import pandas as pd
from pathlib import Path
root_dir = Path(__file__).resolve().parents[3]

sys.path.append(str(root_dir))
from source import MODEL_DIR, FLUX_SAMP_DIR, FVCB_PHI_TABLE
from cobra.sampling import sample
import cobra.io as cio

TAU = 0.0001

def main():
    n_samples = int(sys.argv[1])
    n_cpus = int(sys.argv[2])
    alpha = float(sys.argv[3])
    temp = int(sys.argv[4])

    phi_tab = pd.read_csv(FVCB_PHI_TABLE).loc[:,["Temperature","phi"]].set_index("Temperature").squeeze()
    phi = phi_tab[temp]

    model = cio.load_matlab_model(MODEL_DIR / f"TGEMAdj_{temp}.mat")
    

    expr_le = -1 * model.reactions.arm_RBO_h.flux_expression + model.reactions.arm_RBC_h.flux_expression * phi * (1-TAU)
    constraint_le = model.problem.Constraint(expr_le, ub=0)
    model.add_cons_vars(constraint_le)
    expr_le = + model.reactions.arm_RBO_h.flux_expression - model.reactions.arm_RBC_h.flux_expression * phi * (1+TAU)
    constraint_le = model.problem.Constraint(expr_le, ub=0)
    model.add_cons_vars(constraint_le)
    
    z_star = model.optimize()
    model.reactions.Bio_opt.lower_bound = z_star.objective_value * alpha
    samples = sample(model,
            n = n_samples,
            processes=n_cpus)
    samples.to_csv(FLUX_SAMP_DIR / f"n{n_samples}/fluxes_nsamples.{n_samples}_vbio.{alpha}_{temp}.csv")

if __name__ == "__main__":
    main()