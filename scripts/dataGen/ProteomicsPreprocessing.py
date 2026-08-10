import numpy as np
import pandas as pd
import sys, os
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))
from source import RAW_PROTEOMICS_DATA, PROT_GENES_2_UNIPROT, PROT_ID_2_GENE, PROT_ID_2_DESC, PROT_LONG, PROT_GENES, PROTEOMICS_DATA, MODEL_ENZYME_2_GENE

def main():
    proteomics_27 = pd.read_excel(RAW_PROTEOMICS_DATA,
              sheet_name="standard analysis_27C",
              header=1)
    proteomics_17 = pd.read_excel(RAW_PROTEOMICS_DATA,
                sheet_name="standard analysis_17C",
                header=1)

    d_27 = dict(zip(proteomics_27["PG.ProteinGroups"],proteomics_27["PG.ProteinDescriptions"]))
    d_17 = dict(zip(proteomics_17["PG.ProteinGroups"],proteomics_17["PG.ProteinDescriptions"]))
    d_f = d_27 | d_17 
    pd.Series(d_f).to_csv(PROT_ID_2_DESC)

    if not os.path.exists(PROT_LONG):
        long_17 = format_proteomics(proteomics_17,17)
        long_27 = format_proteomics(proteomics_27,27)
        long_17["Temperature"] = 17
        long_27["Temperature"] = 27
        full_df = pd.concat([long_17,long_27],axis=0)
        full_df.to_csv(PROT_LONG)    
    else:
        full_df = pd.read_csv(PROT_LONG,index_col=0)


    # Create gene mapping for ProteinGroups from IDs (i.e. 'AT1G02930' to 'AT1G02930.1;AT1G02930.2')
    pgs = full_df["PG.ProteinGroups"].str.split(";").explode().unique().astype(str)
    pgs = np.array(np.unique([s[:-2] for s in pgs]))
    np.savetxt(PROT_GENES, pgs, fmt="%s")

    
    # 1. Add the enzyme Uniprot ids to the long proteomics dataframe
    # Map to ProteinIDs (GeneIDs) to ProteinGroup IDs 
    unique_pgs = full_df["PG.ProteinGroups"].unique()
    mapping = (pd.Series(unique_pgs)    # Creates a PG index to gene ID mapping
            .str.split(";")
            .explode()
            .str[:-2]
            .reset_index()
            .rename(columns={"index": "ProteinGroupID", 0: "GeneID"}))

    mapping.set_index("GeneID",inplace=True)
    pg_to_idx = {i: pg for i, pg in enumerate(unique_pgs)} 
    mapping["ProteinGroupID"] = mapping["ProteinGroupID"].map(pg_to_idx)
    mapping.drop_duplicates(inplace=True)
    mapping.to_csv(PROT_ID_2_GENE)

    # 2. Filter all proteomics data that have no enzyme mapping.

    ### ========
    ### EXTERNAL
    ### Use the ID mapping function of UniProt to map the gene names to UniProt entries.
    ### ========

    gene2entry = pd.read_csv(PROT_GENES_2_UNIPROT,sep = "\t").iloc[:,[0,1]]
    ac_enz = pd.read_csv(MODEL_ENZYME_2_GENE,header=None).iloc[:,0]
    rel_gene2entry=gene2entry.loc[gene2entry["Entry"].isin(ac_enz),:]
    rel_gene2entry.columns = ["GeneID","UniprotID"]

    print(f"UniProt ID mapping of the genes within the proteomics data resulted in mapping of {len(rel_gene2entry.loc[:,'UniprotID'].unique())} / 671 proteins in the model.")

    genes_in_data = rel_gene2entry.merge(mapping.reset_index(),on="GeneID",how="inner")
    relevant_data = full_df.loc[full_df["PG.ProteinGroups"].isin(genes_in_data["ProteinGroupID"]),:]
    model_related_proteomics = genes_in_data.merge(right=relevant_data,left_on="ProteinGroupID",right_on="PG.ProteinGroups",how="inner")
    model_related_proteomics.to_csv(PROTEOMICS_DATA)
    return


def format_proteomics(proteomics_27,temp = 27):
    rel_idx = np.where([any([x in c for x in [f"X{temp}","PG."]]) for c in proteomics_27.columns])[0]
    p_27_cut = proteomics_27.iloc[:,rel_idx]
    rel_vars = [c for c in p_27_cut.columns if not "PG." in c]
    rel_vars.extend(["PG.ProteinGroups"])
    p_27_rel = p_27_cut.loc[:,rel_vars].dropna(subset="PG.ProteinGroups")
    p_27_rel.columns = p_27_rel.columns.str.replace(f"X{temp}C.","")
    p_27_rel_long = pd.wide_to_long(
        p_27_rel,
        stubnames=['Sav.0','Bur.0','Cvi.0','Pla.0'],
        i="PG.ProteinGroups",
        j="Replicate",
        sep='.'
        ).reset_index().melt(id_vars = ["PG.ProteinGroups","Replicate"],var_name="Accession",value_name="Relative abundance")  

    abs_vars = [c for c in p_27_cut.columns if "IBAQ" in c]
    abs_vars.extend(["PG.ProteinGroups"])
    p_27_abs = p_27_cut.loc[:,abs_vars].dropna(subset="PG.ProteinGroups")
    p_27_abs.columns = p_27_abs.columns.str.replace(f"X{temp}C.","").str.replace(".PG.IBAQ","")
    p_27_abs_long = pd.wide_to_long(
        p_27_abs,
        stubnames=['Sav.0','Bur.0','Cvi.0','Pla.0'],
        i="PG.ProteinGroups",
        j="Replicate",
        sep='.'
        ).reset_index().melt(id_vars = ["PG.ProteinGroups","Replicate"],var_name="Accession",value_name="Absolute abundance")  


    return pd.merge(left = p_27_rel_long,right = p_27_abs_long,on=["PG.ProteinGroups","Replicate","Accession"])



if __name__ == "__main__":
    main()