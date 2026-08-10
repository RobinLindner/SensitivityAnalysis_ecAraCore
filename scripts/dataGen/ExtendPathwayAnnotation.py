import sys
import numpy as np
import pandas as pd
from pathlib import Path

root_dir = Path(__file__).resolve().parents[3]

sys.path.append(str(root_dir))

from source import RESULT_DIR, MODEL_ENZYME_2_SUBSYSTEM, MODEL_ENZYME_2_SUBSYSTEM_ORI, UNIPROT_ID_2_NAME, MODEL_ENZYME_2_GENE, MAPMAN_ONTOLOGY, PATHWAY_MAPPING_DF


## Output paths
SFILE1_EXCEL = RESULT_DIR / "tables/SensitivityAnalysis/SFile1.xlsx"

## These corrections were manually curated from MapMan and UniProt annotations using the file saved under PATHWAY_MAPPING_DF
corrections = {"F4JJJ3" : ["oxidative phosphorylation"], # External alternative NAD(P)H-ubiquinone oxidoreductase B3, mitochondrial
            "O22048" : ["oxidative phosphorylation"], # Ubiquinol oxidase 1c, mitochondrial
            "O22049" : ["oxidative phosphorylation"], # Ubiquinol oxidase 2, mitochondrial
            "O23913" : ["oxidative phosphorylation"], # Ubiquinol oxidase 1b, mitochondrial
            "O80634" : ["light reactions"], # Photosynthetic NDH subunit of lumenal location 1, chloroplastic
            "O80874" : ["oxidative phosphorylation"], # Internal alternative NAD(P)H-ubiquinone oxidoreductase A2, mitochondrial
            "P09468" : ["light reactions"], # ATP synthase epsilon chain, chloroplastic
            "P0CC32" : ["light reactions"], #NAD(P)H-quinone oxidoreductase subunit 2 A,
            "P0CC33" : ["light reactions"], #NAD(P)H-quinone oxidoreductase subunit 2 B,
            "P17562" : ["methionine synthesis"], # MapMan: amino acid metabolism.synthesis.aspartate family.methionine.S-adenosylmethionine synthetase
            "P19366" : ["light reactions"], # ATP synthase subunit beta, chloroplastic
            "P23686" : ["methionine synthesis"], # MapMan: amino acid metabolism.synthesis.aspartate family.methionine.S-adenosylmethionine synthetase
            "P26288" : ["light reactions"], #NAD(P)H-quinone oxidoreductase chain 4, chloroplastic
            "P26289" : ["light reactions"], #NAD(P)H-quinone oxidoreductase subunit 4L, chloroplastic
            "P28297" : ["gluconeogenesis","glyoxylate cycle"], # MapMan: gluconeogenesis / glyoxylate cycle.isocitrate lyase
            "P42738" : ["phenylalanine synthesis","tyrosine synthesis"],# MapMan: amino acid metabolism.synthesis.aromatic aa.phenylalanine and tyrosine.chorismate mutase
            "P46248" : ["aspartate synthesis"], # MapMan: amino acid metabolism.synthesis.central amino acid metabolism.aspartate.aspartate aminotransferase
            "P46643" : ["aspartate synthesis"], # MapMan: amino acid metabolism.synthesis.central amino acid metabolism.aspartate.aspartate aminotransferase
            "P46644" : ["aspartate synthesis"], # MapMan: amino acid metabolism.synthesis.central amino acid metabolism.aspartate.aspartate aminotransferase
            "P46645" : ["aspartate synthesis"], # MapMan: amino acid metabolism.synthesis.central amino acid metabolism.aspartate.aspartate aminotransferase
            "P46646" : ["aspartate synthesis"], # MapMan: amino acid metabolism.synthesis.central amino acid metabolism.aspartate.aspartate aminotransferase
            "P56751" : ["light reactions"] , # NAD(P)H-quinone oxidoreductase subunit 3, chloroplastic
            "P56752" : ["light reactions"] , # NAD(P)H-quinone oxidoreductase subunit 5, chloroplastic
            "P56753" : ["light reactions"] , # NAD(P)H-quinone oxidoreductase subunit H, chloroplastic
            "P56754" : ["light reactions"] , # NAD(P)H-quinone oxidoreductase subunit J, chloroplastic
            "P56755" : ["light reactions"] , # NAD(P)H-quinone oxidoreductase subunit I, chloroplastic
            "P56756" : ["light reactions"] , # NAD(P)H-quinone oxidoreductase subunit K, chloroplastic
            "P56757" : ["light reactions"] , # ATP synthase subunit alpha, chloroplastic
            "P56758" : ["light reactions"] , # ATP synthase subunit a, chloroplastic
            "P56759" : ["light reactions"] , # ATP synthase subunit b, chloroplastic
            "P56760" : ["light reactions"] , # ATP synthase subunit c, chloroplastic
            "P56771" : ["light reactions"] , # Cytochrome f, petA
            "P56773" : ["light reactions"] , # Cytochrome b6, petB
            "P56774" : ["light reactions"] , # Cytochrome b6-f complex subunit 4, petD
            "P56775" : ["light reactions"] , # Cytochrome b6-f complex subunit 5, petG
            "P56776" : ["light reactions"] , # Cytochrome b6-f complex subunit 6, petL
            "P60112" : ["oxidative phosphorylation"], #MapMan: transport.p- and v-ATPases.H+-transporting two-sector ATPase.subunit C
            "P61039" : ["light reactions"], # Cytochrome b6-f complex subunit 8, petN
            "P83483" : ["oxidative phosphorylation"], #MapMan: mitochondrial electron transport / ATP synthesis.F1-ATPase
            "P83484" : ["oxidative phosphorylation"], #MapMan: mitochondrial electron transport / ATP synthesis.F1-ATPase
            "P92547" : ["oxidative phosphorylation"], # ATP synthase subunit a-2, mitochondrial, ATP6-2
            "P92549" : ["oxidative phosphorylation"], # ATP synthase subunit alpha, mitochondrial, ATPA
            "Q01908" : ["light reactions"], # MapMan: PS.lightreaction.ATP synthase.gamma chain
            "Q01909" : ["light reactions"], # MapMan: PS.lightreaction.ATP synthase.gamma chain
            "Q04613" : ["oxidative phosphorylation"], # ATP synthase protein MI25, mitochondrial
            "Q05758" : ["isoleucine synthesis","valine synthesis"], # MapMan: amino acid metabolism.synthesis.branched chain group.common.ketol-acid reductoisomerase
            "Q1JPL4" : ["oxidative phosphorylation",], # MapMan: mitochondrial electron transport / ATP synthesis.NADH-DH.type II.external
            "Q2V2S7" : ["light reactions"], # MapMan: PS.lightreaction.NADH DH
            "Q37165" : ["light reactions"], # NAD(P)H-quinone oxidoreductase subunit 1, chloroplastic
            "Q39219" : ["oxidative phosphorylation"], # MapMan: mitochondrial electron transport / ATP synthesis.alternative oxidase
            "Q56X52" : ["light reactions"], # MapMan: PS.lightreaction.cyclic electron flow-chlororespiration
            "Q84VQ4" : ["light reactions"], # NAD(P)H-quinone oxidoreductase subunit U, chloroplastic#
            "Q8GWA1" : ["oxidative phosphorylation"], # MapMan: mitochondrial electron transport / ATP synthesis.NADH-DH.type II.internal matrix
            "Q8GXR9" : ["oxidative phosphorylation"], # MapMan: mitochondrial electron transport / ATP synthesis.NADH-DH.type II.mitochondrial
            "Q8GYY0" : [np.nan], # !Conflict between MapMan suggesting involvement in ethylene biosynthesis and Uniprot suggesting the opposite.
            "Q8LEE7" : ["oxidative phosphorylation"], # MapMan: mitochondrial electron transport / ATP synthesis.alternative oxidase
            "Q8RXS1" : ["light reactions"], # Photosynthetic NDH subunit of subcomplex B 4, chloroplastic
            "Q94BV7" : ["oxidative phosphorylation"], # MapMan: mitochondrial electron transport / ATP synthesis.NADH-DH.type II.external
            "Q95695" : ["light reactions"], # NAD(P)H-quinone oxidoreductase subunit 6, chloroplastic
            "Q96250" : ["oxidative phosphorylation"], # MapMan: mitochondrial electron transport / ATP synthesis.F1-ATPase
            "Q96251" : ["oxidative phosphorylation"], # MapMan: mitochondrial electron transport / ATP synthesis.F1-ATPase
            "Q96252" : ["oxidative phosphorylation"], # MapMan: mitochondrial electron transport / ATP synthesis.F1-ATPase
            "Q96253" : ["oxidative phosphorylation"], # MapMan: mitochondrial electron transport / ATP synthesis.F1-ATPase
            "Q9ASS6" : ["light reactions"], # Photosynthetic NDH subunit of lumenal location 5, chloroplastic
            "Q9C544" : ["phenylalanine synthesis","tyrosine synthesis"],# MapMan: amino acid metabolism.synthesis.aromatic aa.phenylalanine and tyrosine.chorismate mutase
            "Q9C5A9" : ["oxidative phosphorylation"], # MapMan:mitochondrial electron transport / ATP synthesis.F1-ATPase
            "Q9CA83" : ["fatty acid synthesis"], # NADP-dependent malic enzyme 4, chloroplastic
            "Q9CA90" : ["photorespiration"], # MapMan: PS.photorespiration.hydroxypyruvate reductase
            "Q9CAC5" : ["light reactions"], # MapMan: PS.lightreaction.NADH DH
            "Q9FG89" : ["light reactions"], # Photosynthetic NDH subunit of subcomplex B 5, chloroplastic
            "Q9FT52" : ["oxidative phosphorylation"], # MapMan: mitochondrial electron transport / ATP synthesis.F1-ATPase
            "Q9LQ10" : [np.nan], # MapMan: hormone metabolism.ethylene.synthesis-degradation.1-aminocyclopropane-1-carboxylate synthase
            "Q9LU21" : ["light reactions"], # MapMan: PS.lightreaction.other electron carrier (ox/red).ferredoxin
            "Q9LUT2" : ["methionine synthesis"], # S-adenosylmethionine synthase 4
            "Q9LVM2" : ["light reactions"], # PS.lightreaction.NADH DH
            "Q9S829" : ["light reactions"], # PS.lightreaction.NADH DH
            "Q9S9N6" : ["light reactions"], # Photosynthetic NDH subunit of subcomplex B 1, chloroplastic
            "Q9SCY3" : ["light reactions"], # Photosynthetic NDH subunit of lumenal location 4, chloroplastic
            "Q9SGH4" : ["light reactions"], # MapMan: PS.lightreaction.photosystem II.PSII polypeptide subunits
            "Q9SIE1" : ["aspartate synthesis"], # MapMan: amino acid metabolism.synthesis.central amino acid metabolism.aspartate.aspartate aminotransferase
            "Q9SIU0" : ["fatty acid synthesis"], # MapMan: TCA / org transformation.other organic acid transformatons.malic
            "Q9SJL8" : ["methionine synthesis"], # MapMan: amino acid metabolism.synthesis.aspartate family.methionine
            "Q9SKT7" : ["oxidative phosphorylation"], # mitochondrial electron transport / ATP synthesis.NADH-DH.type II.external
            "Q9SMS0" : ["light reactions"], # NAD(P)H-quinone oxidoreductase subunit T, chloroplastic
            "Q9SSS9" : ["light reactions"], # MapMan: PS.lightreaction.ATP synthase.delta chain
            "Q9T0A4" : ["light reactions"], # NAD(P)H-quinone oxidoreductase subunit S, chloroplastic
            "Q9XI73" : ["light reactions"], # MapMan: PS.lightreaction.photosystem II.PSII polypeptide subunits
            "Q9ZR03" : ["light reactions"], # MapMan: PS.lightreaction.cytochrome b6/f
            "O48717" : ["light reactions"] # Cytochrome b6f complex subunit (PetM)
            }



# preprocessing of subsystem map to resolve missing values
def main():
    # == STEP 1: CREATE A MAPPING TABLE FOR MANUAL CURATION
    subs_map = pd.read_csv(MODEL_ENZYME_2_SUBSYSTEM_ORI,index_col=0)
    name_map  = pd.read_csv(UNIPROT_ID_2_NAME,index_col=0) 
    subs_map["Name"] = subs_map["Enzyme"].map(name_map.squeeze(),na_action='ignore')
    missing_enzymes = subs_map.loc[subs_map["Subsystem"].isna(),"Enzyme"]
    print(subs_map["Subsystem"].sort_values().unique())

    ## We parse the annotation of MapMan using enzyme gene IDs.
    enz_to_gene = pd.read_csv(MODEL_ENZYME_2_GENE,header = None)
    enz_to_gene.columns = ["Enzyme","Gene"]
    enz_to_gene.set_index("Enzyme",inplace=True)
    ontology = pd.read_csv(MAPMAN_ONTOLOGY, sep="\t")
    enz_gene_map = enz_to_gene.squeeze().to_dict()
    dicti = {"Enzyme":[],"Gene":[],"Ontology":[]}
    for enzyme,gene in enz_gene_map.items():
        idx = ontology["DESCRIPTION"].str.contains(gene.lower())
        if not any(idx):
            dicti["Enzyme"].append(enzyme)
            dicti["Gene"].append(gene)
            dicti["Ontology"].append(np.nan)
            continue
        terms = ontology.loc[idx,"NAME"].str.strip("'").unique()
        for term in terms:
            dicti["Enzyme"].append(enzyme)
            dicti["Gene"].append(gene)
            dicti["Ontology"].append(term)

    pathway_mapping_df = pd.DataFrame(dicti)
    pathway_mapping_df.set_index("Enzyme").loc[missing_enzymes,:]
    pathway_mapping_df.to_csv(PATHWAY_MAPPING_DF)


    # == STEP 2: USE THE MAPPING TABLE TO INFER MANUAL CORRECTIONS
    
    # == STEP 3: APPLY THE CORRECTIONS
    ## Use manually curated corrections from pathway mapping df    
    for enzyme, subsystems in corrections.items():
        # .values[0] to extract the scalar from the array
        name = subs_map.loc[subs_map["Enzyme"] == enzyme, "Name"].values[0]
        
        # drop by index label, not a boolean mask
        subs_map = subs_map.drop(index=subs_map[subs_map["Enzyme"] == enzyme].index)
        
        for sub in subsystems:
            # append a new row as a DataFrame and reset the index
            new_row = pd.DataFrame([[enzyme, sub, name]], columns=subs_map.columns)
            subs_map = pd.concat([subs_map, new_row], ignore_index=True)

    subs_map.to_csv(MODEL_ENZYME_2_SUBSYSTEM)


    dicti = {"UniprotID":[],
                "Protein name":[],
                "Subsystems":[],
                "Newly annotated":[]}
    
    for enzyme in subs_map["Enzyme"].unique():
        name = subs_map.loc[subs_map["Enzyme"] == enzyme, "Name"].values[0]
        subsystems = subs_map.loc[subs_map["Enzyme"] == enzyme, "Subsystem"].dropna().tolist()
        newly_ann = enzyme in corrections.keys()
        if(len(subsystems)<1):
            newly_ann=False
        dicti["UniprotID"].append(enzyme)
        dicti["Protein name"].append(name)
        dicti["Subsystems"].append("; ".join(subsystems))
        dicti["Newly annotated"].append(newly_ann)

    pd.DataFrame(dicti).sort_values("Subsystems").to_excel(SFILE1_EXCEL,index=False)
    

if __name__ == "__main__":
    main()