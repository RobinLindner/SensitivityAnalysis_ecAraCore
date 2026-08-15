from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
SOURCE_DIR = PROJECT_DIR / "src"
DATA_DIR = PROJECT_DIR / 'data'
MODEL_DIR = PROJECT_DIR / 'models'
RESULT_DIR = PROJECT_DIR / 'results'

# create RESULT directory and its subdirectories if it doesn't exist.
(RESULT_DIR / 'figures').mkdir(parents=True, exist_ok=True)
(RESULT_DIR / 'tables').mkdir(parents=True, exist_ok=True)
(RESULT_DIR / 'supplementary').mkdir(parents=True, exist_ok=True)
(RESULT_DIR / 'figures/supplementary').mkdir(parents=True, exist_ok=True)
(DATA_DIR / 'lookup_tables').mkdir(parents=True, exist_ok=True)
(DATA_DIR / 'proteomics').mkdir(parents=True, exist_ok=True)
LOOKUP_TABLES = DATA_DIR / 'lookup_tables'
PROTEOMICS_DIR = DATA_DIR / 'proteomics'

# == PATHS TO MEASUREMENT DATA ==
RAW_PROTEOMICS_DATA = PROTEOMICS_DIR / "A10_PrDa_total_protein_analysis.xlsx"

# == PATHS TO SUPPLEMENTARY INPUT DATA ==
FVCB_PHI_TABLE = LOOKUP_TABLES / "phot_params_across_temp.csv"
MODEL_ID_NAME_MAP = LOOKUP_TABLES / "AraTCore_IDmap.txt"
MODEL_ENZYME_2_GENE = LOOKUP_TABLES / "EnzymeToGeneMap.csv"
MODEL_ENZYME_2_SUBSYSTEM_ORI = LOOKUP_TABLES / "EnzymeToSubsystemMap.csv"
MODEL_ENZYME_2_PMET = LOOKUP_TABLES / "pmet_isoenzyme_map_detailed.csv"
MODEL_ISO_COMPLEX_MAP = LOOKUP_TABLES / "iso_complex_map.csv"
UNIPROT_ID_2_NAME = LOOKUP_TABLES / "upe_names.csv"
UNIPROT_ID_2_DATA = LOOKUP_TABLES / "enzyme_uniprot_data.tsv"
MAPMAN_ONTOLOGY = LOOKUP_TABLES / "Ath_AFFY_STv1.1_TRANSCRIPT_CLUSTER_TAIR10_LOCUS.txt"
GENE_SHORT_NAMES = LOOKUP_TABLES / "gene_name_map.csv"
PROT_GENES_2_UNIPROT = LOOKUP_TABLES / "ProteomicsGenesToUniprot.tsv"


# == PATHS FOR GENERATED DATA / RESULTS == 
## Sensitivity analysis
ESC_DATA_WIDE = RESULT_DIR / "tables/SensitivityAnalysis/wide_SC_10_to_40.csv"
ESC_DATA_COMP_SUMMED = RESULT_DIR / "tables/SensitivityAnalysis/wide_SC_10_to_40_comp_summed.csv"

## Subsystem correction
MODEL_ENZYME_2_SUBSYSTEM = LOOKUP_TABLES / "EnzymeToSubsystemMap_corrected.csv"
PATHWAY_MAPPING_DF = LOOKUP_TABLES/ "Pathway_mapping_supporting_table.csv"

## Clustering
CLUST_RES_DIR = RESULT_DIR / 'tables/Clustering'

## Perturbation
PERT_OUT_DIR = RESULT_DIR / 'tables/PerturbationAnalysis/temperatureSpecific' 
PERT_ESC_WIDE = RESULT_DIR / 'tables/PerturbationAnalysis/Perturbation_ESC_data_wide.csv'
PERT_FLUX_WIDE = RESULT_DIR / 'tables/PerturbationAnalysis/Perturbation_Flux_data_wide.csv'
PERT_CHANGES_WIDE = RESULT_DIR / 'tables/PerturbationAnalysis/Perturbation_Changes_wide.csv'

## flux variability analysis
FVA_DATA_LONG = RESULT_DIR / 'tables/FluxVariabilityAnalysis/fva_95_foo.csv'

## flux sampling 
FLUX_SAMP_DIR = RESULT_DIR / "tables/FluxSampling/"
FLUX_SAMP_WIDE = RESULT_DIR / "tables/FluxSampling/flux_sampling_n.100_a.95_T.10_40.csv"

## pFBA data
PFBA_DATA = RESULT_DIR / "tables/pFBA/pFBA_across_temp.csv"

## Proteomics preprocessing
PROTEOMICS_DATA = PROTEOMICS_DIR / "ModelRelatedProteomics_long.csv"
# ProteinGroupID to ProteinGroup description map
PROT_ID_2_DESC = PROTEOMICS_DIR / "pGroup_pDescription_map.csv"
# Long formated proteomics
PROT_LONG = PROTEOMICS_DIR / "proteomics_long.csv"
# Unique gene names in data
PROT_GENES =  PROTEOMICS_DIR / "AT_geneNames.txt"
# Map between the protein group ID used in the proteomics data and its geneID
PROT_ID_2_GENE = PROTEOMICS_DIR / "ProteinToProteinGroupMap.csv"

PROT_RESULTS_DIR = RESULT_DIR / "tables/RelativeProteomics"


# Main figure directory
FIG_DIR = RESULT_DIR / "figures"

# Supplementary figure directory
SUPP_FIG_DIR = RESULT_DIR / "figures/supplementary/"

# Supplementary data directory
SUPP_RES_DIR = RESULT_DIR / "supplementary/"


