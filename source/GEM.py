import numpy as np
import scipy.sparse as sp
import pandas as pd

from copy import deepcopy
from scipy.io import loadmat
from warnings import warn

#from src.python.model import  main_components_PAM

class GEM:
    """ Python representation of a genome-scale metabolic model (GEM)
    The class was written specifically with Philip's temperature model in mind. Thus, initializing the class relies on
    .mat files with specific field names. If those are not present, the function might fail!
    """

    def __init__(self, model_file, attribute_list={}, is_ec=True, is_pam=False, prot_pfx='prot_', prot_pool='prot_pool'):
        """ Create a GEM object from a .mat model.
        The class stores the following attributes:
        - self.biom_rxn - Biomass reaction ID
        - self.S - Stoichiometric matrix
        - self.c - Objective vector
        - self.b - Right-hand side of FBA constraints (Sv = b)
        - self.lb - Reaction/variable lower bounds
        - self.ub - Reaction/variable upper bounds
        - self.mets - Metabolite IDs
        - self.rxns - Reaction IDs
        - self.comps - Compartments
        - self.genes - Gene IDs
        - self.grRules - Gene-reaction rules
        - self.rxnGeneMat - Reaction-gene matrix
        - self.is_ec - Bool indicating whether the model is enzyme constrained
        - self.is_pam - Bool indicating whether the model is a protein allocation model
        

        For enzyme constrained models the following additional attributes are assigned:
        - self.enzymes - Enzyme IDs
        - self.enzGenes - Enzyme genes
        - self.MWs - Enzyme molecular weights
        - self.prot_pfx - Prefix indicating whether a metabolite is a protein
        - self.prot_pool - ID of the protein pool pseudometabolite

        All remaining fields encoded in the .mat file are all gathered, as a dictionary, in the attribute 'self.other',
        where keys are the field names and values are the field values.

        :param model_file: string, .mat model file
        :param attribute_list: Set, set of additional fields, besides the ones listed above, that should be assigned to attributes
            Use this with care. Depending on the values assigned to the field, this may fail!
        :param is_ec: boolean, set True if the model is an enzyme constrained model
        :param prot_pfx: string, prefix of protein metabolites
        :param protein_pool: string, ID of protein pool pseudometabolite
        """

        self.biom_rxn = None
        self.S = None
        self.c = None
        self.b = None
        self.lb = None
        self.ub = None
        self.mets = None
        self.rxns = None
        self.comps = None
        self.genes = None
        self.grRules = None
        self.rxnGeneMat = None
        self.is_ec = is_ec
        self.is_pam = is_pam
        if is_ec:
            self.enzymes = None
            self.enzGenes = None
            self.MWs = None
            self.prot_pfx = prot_pfx
            self.prot_pool = prot_pool
            self.metabolite_rows = None 
            self.enzyme_rows = None 
            # self.kcat_stats = None

        mat_model = self._read_mat_model(model_file)
        self._assign_attributes(mat_model, attribute_list)
        _, biom_rxn = self.get_biom_rxn(return_id=True)
        self.biom_rxn = biom_rxn[0]
        self.rxns = self.rxns.flatten()     # should be a 1D array
        self.mets = self.mets.flatten()     # should be a 1D array
        if self.is_ec:
            self.enzymes = self.enzymes.flatten()
            self.enzGenes = self.enzGenes.flatten()
            self.enzyme_rows = np.where(np.char.startswith(self.mets,self.prot_pfx))[0]
            self.metabolite_rows = np.where(~np.char.startswith(self.mets, self.prot_pfx) & 
                                            ~np.char.startswith(self.mets, self.prot_pool))[0]

    def copy(self):
        """ Returns a copy of the model """
        return deepcopy(self)

    ### MODEL EXPLORATION ###
    def get_biom_rxn(self, return_id=False):
        """ Returns the index (and ID) of the active biomass reaction
        :param return_id: bool, if True the function also returns the biomass reaction ID
        """
        biom_rxn_idx = np.where(self.c > 0)[0][0]
        biom_rxn_id = self.rxns[biom_rxn_idx]
        if return_id:
            return biom_rxn_idx, biom_rxn_id
        return biom_rxn_idx

    def get_rxn_mets(self, rxn_idx):
        """ Returns the indices of all metabolites participating in a reactions """
        if(sp.issparse(self.S)):
            start = self.S.indptr[rxn_idx]
            end = self.S.indptr[rxn_idx + 1]
            met_idx = self.S.indices[range(start, end)]
        else:
            rxn_vec = self.S[:,rxn_idx]
            met_idx = rxn_vec.nonzero()[0]
        return met_idx

    def set_ptot(self, value):
        self.ub[self.get_rxn_by_id(f"draw_{self.prot_pool}")] = value
        return

    def get_rxn_by_id(self, rxn_id):
        """ Returns the index of a reaction """
        return np.where(self.rxns == rxn_id)[0][0]

    def get_met_by_id(self, met_id):
        """ Returns the index of a metabolite """
        return np.where(self.mets == met_id)[0][0]

    def get_enz_by_id(self,enz_id):
        """ Returns the index of an enzyme """
        return np.where(self.enzymes == enz_id)[0][0]
    
    def get_obj(self):
        """ Returns the indices of all variables in the objective """
        return np.where(self.c != 0)[0]

    def get_catalyzing_enzymes(self, rxn_id):
        """ Returns a list of all enzymes catalyzing a reaction """
        i = self.get_rxn_by_id(rxn_id)
        if sp.issparse(self.S):
            start, end = self.S.indptr[i], self.S.indptr[i + 1]
            met_idx = self.S.indices[range(start, end)]
        else:
            met_idx = np.where(self.S[:,i]!=0)[0]
        enzymes = [met.split('_')[-1] for met in self.mets[met_idx] if met.startswith('prot_')]
        return enzymes

    def get_catalyzed_rxns(self, enzyme):
        """ Returns a list of all reactions catalyzed by a given enzyme """
        i = np.where(self.mets == f"prot_{enzyme}")[0][0]
        S = sp.csr_matrix(self.S)
        start, end = S.indptr[i], S.indptr[i + 1]
        rxn_idx = S.indices[range(start, end)]
        rxns = [self.rxns[j] for j in rxn_idx if np.abs(S[i, j]) != 1]
        return rxns

    def get_kcat(self, enzyme, rxns=None):
        """ Returns the turnover rates of an enzyme, either for all reactions it catalyzes, or only those given. """
        cat_rxns = self.get_catalyzed_rxns(enzyme)
        if rxns is None:
            rxns = cat_rxns
        if len([rxn for rxn in rxns if rxn in cat_rxns]) == 0:
            raise ValueError('Provided reactions are not catalyzed by the given enzyme')
        enzyme = "prot_" + enzyme
        i_enz = list(self.mets).index(enzyme)
        i_rxns = [list(self.rxns).index(rxn) for rxn in rxns]
        kcats = np.array([self.S[i_enz, i_rxn] for i_rxn in i_rxns])
        return -1/kcats

    def get_grRule(self,rxn_id):
        """Returns the grRule of a given reaction ID"""
        rxnIdx = self.get_rxn_by_id(rxn_id)
        return self.grRules[rxnIdx]

    def get_associated_genes(self,rxn_id):
        """Returns an array of gene IDs for a given reaction ID"""
        rxnIdx = self.get_rxn_by_id(rxn_id)
        row = self.rxnGeneMat[rxnIdx, :]       # still a sparse matrix (shape 1 × n)
        geneIdxs = row.nonzero()[1]
        return self.genes[geneIdxs]

    def get_rxn_enz_kcat_mapping(self):
        dicti = {"Enzyme":[],"Reaction":[],"Kcat":[]}
        for enz in self.enzymes:
            rxns = self.get_catalyzed_rxns(enz)
            for rxn in rxns:
                dicti["Enzyme"].append(enz)
                dicti["Reaction"].append(rxn)
                kcat = self.get_kcat(enz,[rxn])[0]
                dicti["Kcat"].append(kcat)

        return(pd.DataFrame(dicti))

    ### MODEL MANIPULATION ###
    def update_bound(self, new_bound, rxn_idx, bound='ub'):
        """ Updates the bound (either upper- or lower-) of a reaction """
        if bound not in ['ub', 'lb']:
            raise ValueError('bound must be either "ub" or "lb"')
        if bound == 'ub':
            self.ub[rxn_idx] = new_bound
        else:
            self.lb[rxn_idx] = new_bound

    def setNitrogenImport(self,type,value):
        '''Limits the Nitrogen import of the model to a given value'''
        if type not in ['NH4', 'NO3']:
            raise ValueError('Imported Nitrogen must be either "NH4" or "NO3"')
        if(type=="NH4"):
            metID = "Im_NH4"
            altID = "Im_NO3"
        elif(type=="NO3"):
            altID = "Im_NH4"
            metID = "Im_NO3"
        self.update_bound(value,self.get_rxn_by_id(metID))
        self.update_bound(0,self.get_rxn_by_id(altID))

    def update_rxn_stoichiometry(self, rxn_idx, met_idx, met_stoich):
        """ Changes the stoichiometric coefficients of a reaction.
        :param rxn_idx: index of the reaction
        :param met_idx: list - Indices of the metabolites partaking in the reaction.
        :param met_stoich: list - Stoichiometric coefficients of the metabolites.
        """
        met_stoich = [i for _, i in sorted(zip(met_idx, met_stoich))]
        met_idx.sort()
        current_met_idx = self.get_rxn_mets(rxn_idx)
        start = self.S.indptr[rxn_idx]
        end = self.S.indptr[rxn_idx + 1]
        if set(current_met_idx) != set(met_idx):
            warn("The current reaction is composed of different metabolites than the one passed!")
            if len(current_met_idx) == len(met_idx):    # different metabolites but same number of metabolites
                self.S.indices[start:end] = met_idx
                self.S.data[start:end] = met_stoich
            else:                                       # different metabolites and different number of metabolites
                size_diff = len(met_idx) - len(current_met_idx)
                self.S.indptr[rxn_idx+1:] += size_diff
                # self.S.indptr = np.hstack([self.S.indptr[:rxn_idx+1], self.S.indptr[rxn_idx+1:] + size_diff])
                self.S.indices = np.hstack([self.S.indices[:start], met_idx, self.S.indices[end:]])
                self.S.data = np.hstack([self.S.data[:start], met_stoich, self.S.data[end:]])
        else:   # same metabolites
            self.S.data[start:end] = met_stoich

    def create_accession_model(self, accession, df_biomass_comp,
                               bio_rxn_id=None, pool_reaction=None,
                               fix_ptot=True, copy=False):
        """ Creates accession specific models by adjusting the biomass composition (stoichiometries of biomass reaction) and
        the total protein amount.

        Note: In the current version of the protein allocation models (PAM) proteins/aminoacids are no longer part of the
        biomass reaction. Thus, those stoichiometries are not adjusted. As a result, the PAM models will not be fully accession
        specific, unless we adjust the model structure.

        :param accession: string, name of the accession (as in the dataframe)
        :param df_biomass_comp: pandas dataframe, contains biomass composition of the different accession, and total protein amounts
        :param bio_rxn_id: string, ID of the biomass reaction to be adjusted. If None is passed, the one stored in the model attribute will be adjusted
        :param pool_reaction: string, ID of the protein pooling reaction (in TGEM 'prot_pool_exchange').
                              If None is passed, it is derived from the ID of the protein pool pseudometabolite stored in the model attribute
        :param fix_ptot: bool, indicating whether total protein amount should be fixed or not. If False ptot becomes unbounded.
        :param copy: bool, indicating whether the current model object should be adjusted, or a copy of it.
        """
        if copy:
            model_out = self.copy()
        else:
            model_out = self

        if bio_rxn_id is None:
            bio_rxn_id = model_out.biom_rxn
        if pool_reaction is None:
            prot_pool_idx = model_out.get_met_by_id(model_out.prot_pool)
            pool_reaction = model_out.rxns[(model_out.S.tocsr()[prot_pool_idx] > 0).indices[0]]

        # update biomass composition
        model_out._update_biomass_rxn(df_biomass_comp, accession, bio_rxn_id)

        # update total protein
        rxn_idx = model_out.get_rxn_by_id(pool_reaction)
        if fix_ptot:
            ptot = df_biomass_comp.loc[accession]['ptot']
            model_out.update_bound(ptot, rxn_idx, 'ub')
        else:
            model_out.update_bound(1000, rxn_idx, 'ub')

        return model_out

    def scale_kcats(self, scaling_factor=10):
        """ Scales ALL! turnover rates by a given factor """
        if not self.is_ec:
            raise ValueError("The model is not an ec model. It does not contain catalytic constants.")

        # get indices of proteins
        prot_idx = [self.get_met_by_id(f"{self.prot_pfx}{enz}") for enz in self.enzymes if enz not in ['ribosome', 'enzyme', 'struct']]

        # for each enzyme, scale kcats
        for i in prot_idx:
            prot_row = self.S.getrow(i)
            kcat_idx = (prot_row < 0).indices
            self.S[i, kcat_idx] *= 1/scaling_factor

    def set_single_kcat(self, how='min'):
        """ For each enzyme, set the turnover rates used in all catalyzed reactions to the same value.
        Possible values are either the min, max, mean or median value over all kcats used for that enzyme.

        :param how: string, determines how the kcat is chosen. Possible values are 'min', 'max', 'mean', 'median'
        """
        if how not in ['min', 'max', 'mean', 'median']:
            raise ValueError("The argument 'how' must be one of the following values: ['min', 'max', 'mean', 'median']")
        
        enz_idx = [k for k, m in enumerate(self.mets) if m.startswith(self.prot_pfx) and m != self.prot_pool]
        S = sp.csr_matrix(self.S)
        enzyme = []
        kcat_min = np.zeros(len(enz_idx))
        kcat_max = np.zeros(len(enz_idx))
        kcat_mean = np.zeros(len(enz_idx))
        kcat_median = np.zeros(len(enz_idx))
        kcat_std = np.zeros(len(enz_idx))
        for k, i in enumerate(enz_idx):
            start = S.indptr[i]
            end = S.indptr[i + 1]
            kcats = -1/S.data[start:end-1]     # assumes the last entry is always the draw-reaction!
            enzyme.append(self.mets[i].split('_')[-1])
            kcat_min[k] = kcats.min()
            kcat_max[k] = kcats.max()
            kcat_std[k] = kcats.std()
            kcat_mean[k] = kcats.mean()
            kcat_median[k] = np.median(kcats)
            if how == 'min':
                S.data[start:end-1] = -1/kcats.min()
            elif how == 'max':
                S.data[start:end-1] = -1/kcats.max()
            elif how == 'mean':
                S.data[start:end-1] = -1/kcats.mean()
            elif how == 'median':
                S.data[start:end-1] = -1/np.median(kcats)
        df_enzyme = pd.DataFrame(
            data={'i': enz_idx, 'enzyme': enzyme, 'min': kcat_min, 'max': kcat_max, 'mean': kcat_mean, 'median': kcat_median},
        )
        self.S = sp.csc_matrix(S)
        return df_enzyme

    def set_new_kcat(self, kcat_dict):
        """ Set new turnover rates for a given set of enzymes.
        (Note: the function will use the same kcat for each catalyzed reaction)
        :param kcat_dict: Dictionary, where keys are enzyme names and values are kcats."""
        S = sp.csr_matrix(self.S)
        for enz, kcat in kcat_dict.items():
            met_idx = self.get_met_by_id(f"prot_{enz}")
            start = S.indptr[met_idx]
            end = S.indptr[met_idx + 1]
            S.data[start:end-1] = -1/kcat
        self.S = sp.csc_matrix(S)


    def get_single_kcat(self,enzyme,how = "min"):
        """ Returns the single kcat of an enzyme, depending on how it was set (min, max, mean, median)
        :param enzyme: string, enzyme ID
        :param how: string, determines how the kcat is chosen. Possible values are 'min', 'max', 'mean', 'median'
        """
        if how not in ['min', 'max', 'mean', 'median']:
            raise ValueError("The argument 'how' must be one of the following values: ['min', 'max', 'mean', 'median']")
        
        enz_idx = self.get_met_by_id(f"{self.prot_pfx}{enzyme}")
        S = sp.csr_matrix(self.S)
        start = S.indptr[enz_idx]
        end = S.indptr[enz_idx + 1]
        kcats = -1/S.data[start:end-1]     # assumes the last entry is always the draw-reaction!
        
        if how == 'min':
            return kcats.min()
        elif how == 'max':
            return kcats.max()
        elif how == 'mean':
            return kcats.mean()
        elif how == 'median':
            return np.median(kcats)
    
    def get_MW(self,enzyme):
        return(-1 * self.S[self.get_met_by_id(self.prot_pool),self.get_rxn_by_id(f"draw_prot_{enzyme}")])

    def get_rxn_equation(self,rxnID):
        rxnIdx = self.get_rxn_by_id(rxnID)
        reaction_vec = self.S[:,rxnIdx]
        sub_idx = np.where(reaction_vec < 0)[0]
        prod_idx = np.where(reaction_vec > 0)[0]
        substrates = " + ".join([f"{abs(self.S[idx,rxnIdx])} {self.mets[idx]}" for idx in sub_idx if idx in self.metabolite_rows])
        products = " + ".join([f"{abs(self.S[idx,rxnIdx])} {self.mets[idx]}" for idx in prod_idx if idx in self.metabolite_rows])
        reaction_string = ""
        if(self.lb[rxnIdx]<0):
            reaction_string = substrates + " <=> " + products
        else:
            reaction_string = substrates + " -> " + products
        return reaction_string
    
    def get_arm_reaction(self,rxnID):
        mets = self.get_rxn_mets(self.get_rxn_by_id(rxnID))
        pmet = [self.mets[idx] for idx in mets if "pmet_" in self.mets[idx]]
        if(len(pmet)<1):
            print("The reaction does not have an arm predecessor.")
            return None
        arm_rxn = [self.rxns[idx] for idx in self.get_met_rxns(pmet[0]) if "arm_" in self.rxns[idx]][0]
        return arm_rxn 


    def get_met_rxns(self,metID):
        met_idx = self.get_met_by_id(metID)
        met_row = self.S[met_idx,:]
        return([rxn for rxn, coeff in enumerate(met_row) if coeff!=0])

    def compartmentalize_enzymes(self,comp_vec):
        """ Returns GEM model with compartmentalized enzymes. For each unique kcat-enzyme-compartment combination 
        a new pseudo-metabolite (enzymatic unit/EU) is introduced. These are then summed independently in 
        enzyme and compartment pools, which in turn are limited by the total available protein (Ptot).
        """
        # NOTE: currently only works by supplying the function with a vector that maps reactions to compartments,
        # since reaction compartmentalization nomenclature is inconsistent in AraTCore and needed to be manually corrected.
        
        # S matrix
        #               Metabolic Reactions      Enzyme draw(mmol/L)   total protein draw(g/gDW) 
        # Metabolites   [metabolic coefficients         0                          0           ]       [0]
        # Enzyme        [ -1/kcat                       1                          0           ]       [0]
        # Ptot          [   0                          -1*MW                       1           ]       [0]



        # S matrix
        #               Metabolic Reactions      EU draw(mmol/L)  Enzyme draw(mmol/L) Compartment draw(mmol/L) total protein draw(g/gDW) 
        # Metabolites   [metabolic coefficients         0               0                   0                       0           ]       [0]
        # EU            [-1/kcat                        1               0                   0                       0           ]       [0]
        # Enzyme        [   0                           -1              1                   0                       0           ]       [0]
        # Compartments  [   0                           -1              0                   1                       0           ] =     [0]
        # Prot Eq       [   0                           0               -1                  1                       0           ]       [0]
        # Ptot          [   0                           0               -1*MW               0                       1           ]       [0]

        if len(comp_vec) != len(self.rxns):
            raise ValueError("The dimension of the compartment vector must match the number of reactions in the model!")

        if not self.is_ec:
            raise ValueError("You can only use this function on ec models!")
        

        # 1.1 Create a mapping for the unique kcat values and localization of enzymes.
        rxn_enz_kcat_df = self.get_rxn_enz_kcat_mapping()
        comp_df = pd.DataFrame({"Reactions":self.rxns,"Compartments":comp_vec})
        rxn_enz_kcat_df = rxn_enz_kcat_df.merge(comp_df,left_on="Reaction",right_on="Reactions")
        
        # 1.2 Define enzymatic units (EU)
        # Enzyme Units are Compartment, Enzyme, and kcat specific.

        EnzymeUnits = rxn_enz_kcat_df.drop(["Reaction","Reactions"],axis=1).drop_duplicates()
        abbrev = np.empty_like(EnzymeUnits["Enzyme"])
        abbrev[EnzymeUnits["Compartments"]=="Chloroplast"] = "_h"
        abbrev[EnzymeUnits["Compartments"]=="Peroxisome"] = "_p"
        abbrev[EnzymeUnits["Compartments"]=="Mitochondrion"] = "_m"
        abbrev[EnzymeUnits["Compartments"]=="IntermembraneSpace"] = "_i"
        abbrev[EnzymeUnits["Compartments"]=="Lumen"] = "_l"
        abbrev[EnzymeUnits["Compartments"]=="Cytosol"] = "_c"
        abbrev[EnzymeUnits["Compartments"]=="NA"] = ""
        EnzymeUnits["Comp_abbrev"] = abbrev
        EnzymeUnits["NewEnzymeID"]= EnzymeUnits["Enzyme"] + EnzymeUnits["Comp_abbrev"]
        EnzymeUnits=self._add_suffix_to_duplicates(EnzymeUnits,"NewEnzymeID")
        
        # 1.3 Assign reactions EU 
        rxn_EU_map = pd.merge(rxn_enz_kcat_df,EnzymeUnits).filter(["Reaction","Enzyme","Kcat","NewEnzymeID","Compartments"])
        
        # 2. Extending the model
        new_GEM = self.copy()

        # 2.1 Extending the model fields
        MetIdxs = [i for i,met in enumerate(self.mets) if "prot_" not in met]
        ProtIdxs = [i for i,met in enumerate(self.mets) if "prot_" in met]
        ProtPoolIdx = ProtIdxs[-1] 
        ProtIdxs = ProtIdxs[:-1] # excluding prot_pool
        RxnIdxs = [i for i,rxn in enumerate(self.rxns) if "draw_" not in rxn]
        ProtPoolExIdx = RxnIdxs[-1]
        RxnIdxs = RxnIdxs[:-1] # excluding "prot_pool_exchange"
        EnzIdxs = [i for i,rxn in enumerate(self.rxns) if "draw_" in rxn]

        Mets = self.mets[MetIdxs]
        Prots = self.mets[ProtIdxs]
        Rxns = self.rxns[RxnIdxs]
        Enz = self.rxns[EnzIdxs]
        MW = - self.S.todense()[ProtPoolIdx,EnzIdxs].T


        EU = np.array(["EU_" + eu for eu in rxn_EU_map["NewEnzymeID"].unique()])
        EU_draw = np.array(["draw_EU_" + eu for eu in rxn_EU_map["NewEnzymeID"].unique()])


        Comp = np.array(["comp_" + c for c in self.comps]).reshape(-1,)
        Comp_draw = np.array(["draw_comp_" + c for c in self.comps]).reshape(-1,)

        Comp = np.concat((Comp,["comp_NA"]))
        Comp_draw = np.concat((Comp_draw,["draw_comp_NA"]))

        new_GEM.mets = np.concat((Mets,EU,Prots,Comp,["prot_equality"],["prot_pool"]))
        new_GEM.rxns = np.concat((Rxns,EU_draw,Enz,Comp_draw,["prot_pool_exchange"]))

        new_GEM.b = np.zeros((len(new_GEM.mets),)) 
        new_GEM.c = np.zeros((len(new_GEM.rxns),)) 
        new_GEM.biom_rxn = np.where(new_GEM.rxns == self.biom_rxn)[0][0]
        new_GEM.c[new_GEM.biom_rxn] = 1

        new_GEM.lb = np.zeros((len(new_GEM.rxns),)) 
        new_GEM.lb[:len(Mets)] = self.lb[MetIdxs].reshape(-1,)
        new_GEM.lb[-1] = self.lb[self.get_rxn_by_id("prot_pool_exchange")]

        new_GEM.ub = np.ones((len(new_GEM.rxns),))*1000 
        new_GEM.ub[:len(Mets)] = self.ub[MetIdxs].reshape(-1,)
        new_GEM.ub[-1] = self.ub[self.get_rxn_by_id("prot_pool_exchange")]


        # look-up dictionaries for creating the S matrix.
        kcat_lookup = dict(zip(rxn_EU_map["Reaction"]+"_"+rxn_EU_map["NewEnzymeID"],rxn_EU_map["Kcat"]))
        enzyme_lookup = dict(zip(rxn_EU_map["NewEnzymeID"],rxn_EU_map["Enzyme"]))
        mw_lookup = dict(zip(Enz,MW))
        comp_lookup = dict(zip(rxn_EU_map["NewEnzymeID"],rxn_EU_map["Compartments"]))


        new_S = np.zeros((len(new_GEM.mets),len(new_GEM.rxns)))
        
        print("Constructing S matrix, might take up to 2 minutes.")
        for i, m in enumerate(new_GEM.mets):
            
            for j, r in enumerate(new_GEM.rxns):
                
                if m in Mets:
                    # case: reaction stoichiometry
                    if r in Rxns :
                        m_idx = np.where(self.mets==m)[0][0]
                        r_idx = np.where(self.rxns==r)[0][0]
                        new_S[i,j] = self.S[m_idx,r_idx]

                elif m in EU:
                    # case: turnover numbers
                    if r in Rxns:
                        eu = m.removeprefix("EU_")
                        if r+"_"+eu in kcat_lookup:
                            kcat=kcat_lookup[r+"_"+eu]
                            new_S[i,j] = -1 / kcat 
                    # case: EU draw variables
                    elif r in EU_draw:
                        if m == r.removeprefix("draw_"):
                            new_S[i,j] = 1
                
                elif m in Prots:
                    # case: EU draw depletes enzyme pool
                    if r in EU_draw:
                        enz = m.removeprefix("prot_")
                        if enzyme_lookup[r.removeprefix("draw_EU_")] == enz:
                            mw = mw_lookup["draw_prot_"+enz].item()
                            new_S[i,j] = -1 

                    # case: Enzyme draw supplies enzyme pool
                    elif r in Enz:
                        if m == r.removeprefix("draw_"):
                            new_S[i,j] = 1
                
                elif m in Comp:
                    # case EU draw depletes compartment pool
                    if r in EU_draw:
                        comp = self._translateCompAbbrev(m.removeprefix("comp_"))
                        if comp_lookup[r.removeprefix("draw_EU_")]==comp:
                            enzyme = enzyme_lookup[r.removeprefix("draw_EU_")]
                            mw = mw_lookup["draw_prot_"+enzyme].item()
                            new_S[i,j] = -1 

                    # case: Compartment draw supplies compartment pool
                    elif r in Comp_draw:
                        if m == r.removeprefix("draw_"):
                            new_S[i,j] = 1

                elif m == "prot_equality":
                    if r in Enz:
                        new_S[i,j] = -1
                    elif r in Comp_draw:
                        new_S[i,j] = 1
                elif m == "prot_pool":
                    if r in Enz:
                        mw = mw_lookup[r].item()
                        new_S[i,j] = -1 * mw
                    elif r == "prot_pool_exchange":
                        new_S[i,j] = 1
        
        print("Finished S construction.")
        new_GEM.S = new_S

        return new_GEM








    ### PRIVATE METHODS ###
    def _assign_attributes(self, mat_model, attributes):
        """ Assign selected struct fields from the .mat file to corresponding class attributes. """
        # FIXME: Probably rewrite at a later point. This function is horrible!

        # check if all mandatory attributes are present in the mat-model
        mandatory_attributes = {'S', 'b', 'c', 'lb', 'ub', 'mets', 'rxns', 'comps', 'genes', 'grRules', 'rxnGeneMat'}
        if self.is_ec:
            mandatory_attributes = mandatory_attributes.union({'enzymes', 'enzGenes', 'MWs'})
        attr_names = set(mat_model.dtype.names)
        if len(mandatory_attributes.intersection(attr_names)) < len(mandatory_attributes):
            missing_attr = mandatory_attributes.difference(attr_names)
            raise ValueError(f'Some mandatory attributes are missing from model: {missing_attr}')

        # check if all attributes that should be assigned are in the mat-model
        attr_to_assign = mandatory_attributes.union(attributes)
        if len(attr_to_assign.intersection(attr_names)) < len(attr_to_assign):
            attrNA = attr_to_assign.difference(attr_names)
            warn(f'Attribute(s) {attrNA} not present in the passed model.')
            attr_to_assign = attr_to_assign.intersection(attr_names)

        for attr in attr_to_assign:
            # make sure numerical values are stored as floats
            dtype = None
            if attr in ['S', 'rxnGeneMat', 'b', 'c', 'lb', 'ub']:
                dtype = float

            attr_value = mat_model[attr].flatten()[0]
            if attr_value.dtype == np.object_:  # the array is a 2-d array, containing numpy array objects (effectively it's a 3d array!)
                try:
                    attr_value = np.array(np.vstack(attr_value.flatten()), dtype=dtype)
                except:
                    # some of the array elements are empty lists
                    attr_value = np.array([val if len(val) > 0 else [''] for val in attr_value.flatten()], dtype=dtype)
            else:
                if dtype is not None:
                    attr_value = attr_value.astype(dtype)

            self.__setattr__(attr, attr_value)
 
        # all the remaining attributes of the mat-model are gathered in a dictionary and assigned to the 'other' attribute
        attr_not_assigned = attr_names.difference(attr_to_assign)
        self.other = {}
        for attr in attr_not_assigned:
            self.other[attr] = mat_model[attr].flatten()[0]

    def _read_mat_model(self, model_file):
        """ Read in .mat file """
        mat = loadmat(model_file)
        key = [key for key in mat.keys() if not key.startswith('__')][0]
        return mat[key]

    
    def _add_suffix_to_duplicates(self,df, column_name):
            """
            Appends a sequential numeric suffix (_0, _1, ...) to duplicate values 
            in a specified DataFrame column to make all entries unique.

            Args:
                df (pd.DataFrame): The input DataFrame.
                column_name (str): The name of the column to process.

            Returns:
                pd.DataFrame: The DataFrame with unique values in the specified column.
            """
            # Group by the column and calculate a cumulative count for each value
            # cumcount() starts from 0 for each group
            counts = df.groupby(column_name).cumcount()
            
            # Create a mask for all duplicated values (including the first occurrence)
            is_duplicate = df[column_name].duplicated(keep=False)
            
            # Apply the suffix only to the duplicated values
            # For non-duplicates, the original value is kept
            df.loc[is_duplicate, column_name] = df.loc[is_duplicate, column_name].astype(str) + '_' + counts.loc[is_duplicate].astype(int).astype(str)
            
            return df
    
    def _translateCompAbbrev(self,abbrev):
        if(abbrev=="h"):
            return "Chloroplast"
        elif(abbrev=="c"):
            return "Cytosol"
        if(abbrev=="m"):
            return "Mitochondrion"
        if(abbrev=="p"):
            return "Peroxisome"
        if(abbrev=="i"):
            return "IntermembraneSpace"
        if(abbrev=="l"):
            return "Lumen"
        else:
            return "NA"
