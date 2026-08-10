import numpy as np
import pandas as pd
from pathlib import Path
import sys,io

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))

from source import MODEL_DIR, PFBA_DATA
from source.OptimizationProblem import OptimizationProblem
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


def main():
    flux_vecs = []
    for temp in np.arange(10,41):
        model = GEM(MODEL_DIR / f"TGEMAdj_{temp}.mat")
        print(f"Performing pFBA at {temp}°C")
        with silence():
            op = OptimizationProblem(model)
            op.create_pfba_problem(0.95,flux_option=1)
            op.pfba.optimize()
        flux_vecs.append(pd.Series({v.VarName: v.X for v in op.pfba.getVars()}))
    full_df = pd.concat(flux_vecs,axis=1)
    print(full_df.head())
    full_df.columns = np.arange(10,41,1)
    pFBA_long = full_df.reset_index(names="Variable").melt(id_vars="Variable",var_name="Temperature",value_name="Flux")
    pFBA_long.to_csv(PFBA_DATA)


if __name__ == "__main__":
    main()