import numpy as np
import pandas as pd
from pathlib import Path
import sys, ast

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))

from source import MODEL_DIR, ESC_DATA_WIDE, ISO_COMPLEX_MAP, MODEL_ENZYME_2_PMET, ESC_DATA_COMP_SUMMED
from source.GEM import GEM
from collections import Counter



def main():

    SC_wide = pd.read_csv(ESC_DATA_WIDE,index_col=0)

    # #### Group isoenzymes into single enzyme complex.
    # for all pmet reactions , identify the subsequent enzymes 
    # !ISSUE: This includes obligate subunits of heteroenzymes, i.e RbcL and subunits!
    #   Only the actual isoenzymes (subunits in case of RuBisCO) should be summed. 
    model = GEM(MODEL_DIR / "TGEMAdj_20.mat")
    dicti = {"Pmet":[],
            "Isoenzymes":[]
            }
    pmets = [met for met in model.mets if "pmet_" in met]
    dfs = list()
    for pmet in pmets:
        # A pseudometabolite connects an arm reaction to isoreactions
        dicti["Pmet"].append(pmet)
        # Iso reactions are all downstream reactions (i.e. connected to pseudo metabolite but not arm) 
        isorxns = [model.rxns[rxnidx] for rxnidx in model.get_met_rxns(str(pmet)) if not "arm" in model.rxns[rxnidx]]
        # Build dictionary mapping the isoreactions to the catalyzing enzymes
        isorxn_enzymes = {rxn : [enz for enz in model.get_catalyzing_enzymes(rxn)] for rxn in isorxns}

        all_enzymes = [s for subunits in isorxn_enzymes.values() for s in subunits]
        subunit_counts = Counter(all_enzymes)
        n_isoreactions=len(isorxns)

        # enzymes shared by all iso reactions
        obligate = [s for s,count in subunit_counts.items() if count >= n_isoreactions]

        # enzymes shared by more than one iso reaction
        shared = [s for s, count in subunit_counts.items() if (count > 1) & (count < n_isoreactions)]

        # enzymes unique to one iso reaction
        unique = [s for s,count in subunit_counts.items() if count == 1]

        # print(f"Pmet: {pmet};\t Isorxns: {n_isoreactions};\t Obligate: {len(obligate)};\t Shared: {len(shared)};\t Unique: {len(unique)}")

        dicti["Isoenzymes"].append(unique)

        t_df = pd.DataFrame({
            "Pmet":pmet,
            "Isorxns": [key for key,value in isorxn_enzymes.items() for v in value],
            "Enzymes" : [v for key,value in isorxn_enzymes.items() for v in value],
            "Type": "Unique"
        })
        t_df.loc[t_df["Enzymes"].isin(obligate),"Type"] = "Obligate"
        t_df.loc[t_df["Enzymes"].isin(shared),"Type"] = "Shared"
        dfs.append(t_df)



    Pmet_Isoenzyme_map = pd.DataFrame(dicti)
    #Pmet_Isoenzyme_map.to_csv(DATA_DIR / "sEnz_datasets/pmet_isoenzyme_map.csv")

    Pmet_Isoenzyme_map_detailed = pd.concat(dfs).reset_index().drop(columns="index")
    Pmet_Isoenzyme_map_detailed.to_csv(MODEL_ENZYME_2_PMET)


    ## Isoenzymes which can be substituted/are unique in the set of isoreactions 
    # can be further consolidated by combining 'pseudo metabolites' sharing identical isoenzyme sets
    Pmet_Isoenzyme_map.set_index("Pmet",inplace=True)


    n = Pmet_Isoenzyme_map.shape[0]
    Pmet_Isoenzyme_map.loc[:,"Isoenzymes"] = [frozenset(lst) for lst in Pmet_Isoenzyme_map["Isoenzymes"]]
    n_u = len(Pmet_Isoenzyme_map["Isoenzymes"].unique())
    print()
    print(f"There are {n} pseudometabolites representing enzyme complexes in the model.")
    print(f"Within these there are {n_u} unique sets of enzymes.")
    print(f"This means that {np.round((n-n_u)/n*100)}% can be further consolidated, as there exists an alternative with the same sensitivity leading to redundancy.")
    print()

    first_occurrences = Pmet_Isoenzyme_map["Isoenzymes"].reset_index().drop_duplicates(subset=Pmet_Isoenzyme_map["Isoenzymes"].name, keep="first")["Pmet"]
    unique_iso = pd.Series(
        first_occurrences.values,index=Pmet_Isoenzyme_map["Isoenzymes"].unique()
        )

    Pmet_Isoenzyme_map["ComplexID"] = Pmet_Isoenzyme_map["Isoenzymes"].map(unique_iso) 
    Pmet_Isoenzyme_map = Pmet_Isoenzyme_map.reset_index().loc[:,["ComplexID","Pmet","Isoenzymes"]]
    #Pmet_Isoenzyme_map.to_csv(DATA_DIR / "sEnz_datasets/pmet_isoenzyme_map.csv")


    ## Create a long form map that allows quick access of protein complexes from enzyme ids.
    Pmet_Isoenzyme_map_long = Pmet_Isoenzyme_map_detailed[Pmet_Isoenzyme_map_detailed["Type"]=="Unique"].drop(columns=["Isorxns","Type"]).drop_duplicates()
    Pmet_Isoenzyme_map_long.columns = ["Pmet","Isoenzymes"]
    Pmet_Isoenzyme_map_long["ComplexID"] = Pmet_Isoenzyme_map_long["Pmet"].map(Pmet_Isoenzyme_map.set_index("Pmet")["ComplexID"])
    Pmet_Isoenzyme_map_long = Pmet_Isoenzyme_map_long.loc[:,["Isoenzymes","ComplexID","Pmet"]].drop_duplicates()
    Pmet_Isoenzyme_map_long.to_csv(ISO_COMPLEX_MAP)


    Pmet_Isoenzyme_map.drop_duplicates(subset="ComplexID",keep="first",inplace=True)
    Pmet_Isoenzyme_map.set_index("Pmet",inplace=True)

    covered_enzymes = list()
    SC_new = SC_wide.copy()
    for pmet in Pmet_Isoenzyme_map.index:
        isoenzymes = list(Pmet_Isoenzyme_map.loc[pmet,"Isoenzymes"])
        pmet_sum = SC_new.loc[isoenzymes,:].sum(axis=0)
        SC_new.loc[pmet,:] = pmet_sum
        covered_enzymes.extend(isoenzymes)
    SC_new = SC_new.drop(set(covered_enzymes),axis=0)

    print(f"Of the 671 enzymes in the model, {len(set(covered_enzymes))} are unique to a single isoreaction and can be substituted. \nWe sum these isoenzymes because small differences in metabolic efficiency can result in an on-off behavior \n for the sensitivity where only one alternate isoenzyme is active at any time.\nBy computing the sum of these we get a more stable estimate.")
    print(f"This leaves {671 - len(set(covered_enzymes))} enzymes as independent in the model.")
    SC_new.to_csv(ESC_DATA_COMP_SUMMED)
    print()


if __name__ == "__main__":
    main()

