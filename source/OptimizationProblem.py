import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from copy import deepcopy

#from src.python.parameters import PAM_dicti
#from src.python.optimization.pam_optimization import add_pam_constraints, add_mol_crowding_constraint

class OptimizationProblem:

    def __init__(self, gem):
        # FIXME: Storing the model object as an attribute is not a good idea. I should change that
        self.gem = gem.copy()
        self.is_ec = gem.is_ec
        self.fba = None
        self.pfba = None
        self.pam = None
        self.room = None
        self.renzo = None

    def create_fba_problem(self):
        """ Creates the FBA problem for the stored GEM.
        If the GEM is enzyme-constrained, the enzyme constraints are added as inequality constraints (i.e. 1/kcat*v <= E)
        """
        n_mets, n_rxns = self.gem.S.shape

        # create gurobipy model
        self.fba = gp.Model()
        self.fba.setParam('NumericFocus', 3)
        # add variables
        varnames = np.reshape(self.gem.rxns, (-1, 1))
        
        mvars = self.fba.addMVar(shape=self.gem.lb.reshape(-1,1).shape, lb=self.gem.lb.reshape(-1,1), ub=self.gem.ub.reshape(-1,1), obj=self.gem.c.reshape(-1,1),
                                 name=varnames, vtype=gp.GRB.CONTINUOUS)
        mvars = mvars.reshape(-1)

        # add constraints
        if self.is_ec:
            # get protein indices
            prot_idx = [self.gem.get_met_by_id(f"{self.gem.prot_pfx}{enz}") for enz in self.gem.enzymes]
            prot_pool_idx = self.gem.get_met_by_id(f"{self.gem.prot_pool}")
            met_idx = [i for i in range(n_mets) if i not in [prot_idx,prot_pool_idx]]
            
            # equality constraints
            Aeq = self.gem.S
            beq = self.gem.b.reshape((-1,))
            self.fba.addMConstr(A=Aeq, x=mvars, b=beq, sense=gp.GRB.EQUAL, name=self.gem.mets)
            
            ''' DEPRECATED: Everything should be handled as equalities, see GECKO supplements


            # inequality constraints (enzyme capacity)
            Aeq2 = self.gem.S[prot_idx, :]
            beq2 = self.gem.b[prot_idx].reshape((-1,))
            self.fba.addMConstr(A=Aeq2, x=mvars, b=beq2, sense=gp.GRB.EQUAL, name=self.gem.mets[prot_idx])
            
            prot_pool_row = self.gem.S[prot_pool_idx, :].reshape(1,-1)
            pTot = self.gem.b[prot_pool_idx].reshape((-1,))
            self.fba.addMConstr(A=prot_pool_row, x=mvars, b=pTot, sense=gp.GRB.GREATER_EQUAL, name=self.gem.mets[prot_pool_idx])
            '''
        else:
            b = self.gem.b.reshape((-1,))
            self.fba.addMConstr(A=self.gem.S, x=mvars, sense=gp.GRB.EQUAL, b=b, name=self.gem.mets)

        # set objective
        self.fba.update()
        var_biomass = self.fba.getVarByName(self.gem.biom_rxn)
        self.fba.setObjective(var_biomass, GRB.MAXIMIZE)

        # update model
        self.fba.update()

    def create_pfba_problem(self, alpha=0.95, gene_option=0, flux_option = 1 ,create_fba=True, use_pam=False, solver_parameters={'OutputFlag': 0}):
        """ Creates the parsimonious FBA problem (pFBA), which minimizes the sum of all (or some) fluxes for a given growth rate.

        :param alpha: float, determines the amount of suboptimality. Biomass will be fixed as v_biomass >= alpha*z*
        :param gene_option: int (0 or 1), determines which reaction fluxes to minimize.
                            0 - minimize the sum of all fluxes
                            1 - minimize the sum of all fluxes that are controlled by some gene
        :param flux_option: int (0,1,2), determines which types of fluxes to minimize.
                            0 - minimize the sum of all fluxes
                            1 - minimize the sum of all metabolic fluxes 
                            0 - minimize the sum of all enzyme abundance fluxes
        :param create_fba: bool, set to True to create a new FBA problem.
        :param use_pam: bool, set to True to run pFBA on a protein allocation model
        :param solver_parameters: dictionary of solver parameters (keys - parameter names, values - parameter values)
        """
        if gene_option not in [0, 1]:
            raise ValueError("The argument 'gene_option' must be 0 or 1")

        # create and solve FBA/PAM problem if necessary
        if use_pam:
            if self.pam is None:
                raise AttributeError('You must create the protein allocation model (PAM) first')
            op = self.pam
        else:
            if self.fba is None or create_fba:
                self.create_fba_problem()
            op = self.fba

        for p_name, p_value in solver_parameters.items():
            op.setParam(p_name, p_value)
            
        if op.status != GRB.OPTIMAL:
            op.optimize()
            if op.status != GRB.OPTIMAL:
                raise Exception("Optimization failed")

        # get objective and objective function from solved FBA
        self.pfba = op.copy()
        obj_fun = op.Obj
        z_star = op.ObjVal
        sense = op.ModelSense

        # add FBA objective as constraint
        vars = self.pfba.getVars()
        expr = gp.LinExpr(obj_fun, vars)
        if sense == 1:
            self.pfba.addLConstr(expr, GRB.LESS_EQUAL, z_star*(2-alpha), 'prev_objective')
        else:
            self.pfba.addLConstr(expr, GRB.GREATER_EQUAL, z_star*alpha, 'prev_objective')

        # add auxiliary absolute value variables
        if flux_option == 0:
            vars = self.pfba.getVars()
        elif flux_option == 1:
            vars = [v for v in self.pfba.getVars() if not (("draw_prot_" in v.VarName) | ("prot_pool_" in v.VarName))]
        else:
            vars = [v for v in self.pfba.getVars() if "draw_prot_" in v.VarName]


        n_vars = len(vars)
        abs_vars = self.pfba.addVars(n_vars, lb=0, name="abs_aux")
        self.pfba.update()

        for i, var in enumerate(vars):
            self.pfba.addConstr(abs_vars[i] >= var,  f"abs_pos_{i}")
            self.pfba.addConstr(abs_vars[i] >= -var, f"abs_neg_{i}")

        '''
        # build coefficient mask c 
        c = np.zeros(n_vars)
        if gene_option == 0:
            c = np.ones(n_vars)
        elif gene_option == 1:
            rxn_idx = np.unique(self.gem.rxnGeneMat.indices)
            c[rxn_idx] = 1

        if flux_option == 1:
            c = np.zeros(n_vars)
            flux_idx = np.invert(["prot_pool" in var.VarName or "draw_prot" in var.VarName 
                                for var in vars])
            c[flux_idx] = 1
        elif flux_option == 2:
            c = np.zeros(n_vars)
            flux_idx = ["draw_prot" in var.VarName for var in vars]
            c[flux_idx] = 1
        '''

        print(f"Optimal baseline biomass flux: {z_star}")
        print(f"pFBA minimal biomass bound: {z_star*alpha}")
        print(f"Fluxes being minimized: {n_vars}")

        # minimize sum of absolute fluxes using aux variables
        expr = gp.LinExpr(np.ones((n_vars,)), [abs_vars[i] for i in range(n_vars)])
        for p_name, p_value in solver_parameters.items():
            self.pfba.setParam(p_name, p_value)
        self.pfba.setObjective(expr, GRB.MINIMIZE)

    def run_fva(self, target_vars=None, alpha=0.95, create_fba=True, verbose = True, solver_parameters={'OutputFlag': 0}):
        """ Run a flux variability analysis for all/some reactions to determine their operational range.

        :param target_vars: list of str, list of reaction ID for which to compute the operational ranges
        :param alpha: float, determines the amount of suboptimality. Biomass will be fixed as v_biomass >= alpha*z*
        :param create_fba: set True to create a new FBA problem
        :param solver_parameters: dictionary of solver parameters (keys - parameter names, values - parameter values)
        """

        # create and solve fba problem if necessary
        if self.fba is None or create_fba:
            self.create_fba_problem()
        for p_name, p_value in solver_parameters.items():
            self.fba.setParam(p_name, p_value)
        if self.fba.status != GRB.OPTIMAL:
            self.fba.optimize()
            if self.fba.status != GRB.OPTIMAL:
                raise Exception("Optimization failed")

        z_star = self.fba.ObjVal

        # create FVA problem
        fva = self.fba.copy()
        prev_obj = fva.getObjective()
        sense = fva.ModelSense
        if sense == 1:      # the previous objective was a minimization problem
            fva.addConstr(prev_obj <= (2-alpha)*z_star, name='fix_biomass')
        else:               # the previous objective was a maximization problem
            fva.addConstr(prev_obj >= alpha*z_star, name='fix_biomass')

        if target_vars is None:
            variables = fva.getVars()
        else:
            variables = [fva.getVarByName(vname) for vname in target_vars]

        nvars = len(variables)
        fva_dicti = {'var': [], 'min': [], 'max': []}
        for i, var in enumerate(variables):
            if(verbose):
                print(f"Variable {i+1}/{nvars}: {var.VarName}")
            # minimize variable
            fva.setObjective(var, GRB.MINIMIZE)
            fva.optimize()
            z_min = fva.ObjVal
            # maximize variable
            fva.setObjective(var, GRB.MAXIMIZE)
            fva.optimize()
            z_max = fva.ObjVal
            # add results to dicti
            fva_dicti['var'].append(var.VarName)
            fva_dicti['min'].append(z_min)
            fva_dicti['max'].append(z_max)

        df_fva = pd.DataFrame(fva_dicti)
        return df_fva


    def create_RENZO_FVA_problem(self, secondary_gem, enzyme_abundance_ratios, pTot_ratio, growth_ratio ,bio_delta = 0.85, FVA_epsilon = 1e-5):
        """ Creates a RENZO problem for the stored GEM and a secondary GEM.
        Both gems have to be enzyme constrained for this function
        """
        if(not np.array_equal(self.gem.rxns,secondary_gem.rxns)):
            raise AttributeError('Models do not contain the same reaction identifiers!')

        if(not np.array_equal(self.gem.enzymes,secondary_gem.enzymes)):
            raise AttributeError('Models do not contain the same enzyme identifiers!')

        prot_ex_idx = self.gem.get_rxn_by_id("prot_pool_exchange")
        secondary_gem.ub[prot_ex_idx] = pTot_ratio * self.gem.ub[prot_ex_idx]

        bio_idx = self.gem.get_rxn_by_id(self.gem.biom_rxn)

        renzo = gp.Model()
        renzo.setParam('NumericFocus', 3)

        ## Variables
        # add variables of gem 1
        varnames_1 = np.reshape(self.gem.rxns, (-1, 1))

        mvars = renzo.addMVar(shape=self.gem.lb.reshape(-1,1).shape, lb=self.gem.lb.reshape(-1,1), ub=self.gem.ub.reshape(-1,1), obj=self.gem.c.reshape(-1,1),
                                name=varnames_1, vtype=gp.GRB.CONTINUOUS)
        mvars_1 = mvars.reshape(-1)

        # add variables of gem 2
        varnames_2 = np.reshape(secondary_gem.rxns+"_mut", (-1, 1))

        mvars = renzo.addMVar(shape=secondary_gem.lb.reshape(-1,1).shape, lb=secondary_gem.lb.reshape(-1,1), ub=secondary_gem.ub.reshape(-1,1), obj=secondary_gem.c.reshape(-1,1),
                                name=varnames_2, vtype=gp.GRB.CONTINUOUS)
        mvars_2 = mvars.reshape(-1)

        # add variables for deviations in the relative proteomics
        theta = np.reshape("theta_" + self.gem.enzymes, (-1, 1))
                    
        tvars = renzo.addMVar(shape=theta.shape, lb=np.zeros(theta.shape), ub=np.ones(theta.shape)*1000, obj=np.zeros(theta.shape),
                                name=theta, vtype=gp.GRB.CONTINUOUS)


        ## Constraints
        # metabolic constraints model 1
        Aeq_1 = self.gem.S
        beq_1 = self.gem.b.reshape((-1,))
        cons_1 = np.reshape(self.gem.mets +"_mc1", (-1, ))
        renzo.addMConstr(A=Aeq_1, x=mvars_1, b=beq_1, sense=gp.GRB.EQUAL, name=cons_1)

        # metabolic constraints model 2
        Aeq_2 = secondary_gem.S
        beq_2 = secondary_gem.b.reshape((-1,))
        cons_2 = np.reshape(secondary_gem.mets +"_mc2", (-1, ))
        renzo.addMConstr(A=Aeq_2, x=mvars_2, b=beq_2, sense=gp.GRB.EQUAL, name=cons_2)

        # relative proteomics constraints
        #          e_wt     e_m     theta
        # Aineq = [ f       -1      -1  ]   ≤   0
        #         [ -f      1       -1  ]   ≤   0

        e_wt = mvars_1[[[self.gem.get_rxn_by_id("draw_prot_"+e) for e in self.gem.enzymes]]]
        e_m = mvars_2[[[secondary_gem.get_rxn_by_id("draw_prot_"+e) for e in secondary_gem.enzymes]]]
        print(f"Wildtype {e_wt.shape}, Mutant: {e_m.shape}, Theta {tvars.shape}, Ratios: {enzyme_abundance_ratios.shape}")

        tvars_1d = tvars.reshape(-1)
        renzo.addConstr(enzyme_abundance_ratios * e_wt- e_m - tvars_1d <= 0,name="theta")
        renzo.addConstr(-enzyme_abundance_ratios * e_wt + e_m - tvars_1d <= 0,name="theta")

        renzo.setObjective(mvars_1[bio_idx],gp.GRB.MAXIMIZE)

        renzo.optimize()

        z_star = renzo.ObjVal
        wt_bio = mvars_1[bio_idx]
        mut_bio = mvars_2[bio_idx]

        # Biomass distance constraint
        renzo.addConstr( - wt_bio <= -1 * (bio_delta * z_star),name="Biomass distance from wt")

        # Biomass ratio constraint
        renzo.addConstr( mut_bio - growth_ratio * wt_bio == 0, "Biomass ratio to wt")

        renzo.setObjective(tvars.sum(), gp.GRB.MINIMIZE)

        renzo.update()
        print(f"DEBUG: obj{renzo.getObjective()}")
        renzo.optimize()
        renzo.ObjVal
        ## FVA step

        class IterativeStats:
            def __init__(self, n_vars):
                self.n = 0
                self.mean = np.zeros(n_vars)
                self.M2 = np.zeros(n_vars)   # sum of squared deviations

            def update(self, x):
                self.n += 1
                delta = x - self.mean
                self.mean += delta / self.n
                delta2 = x - self.mean
                self.M2 += delta * delta2

            @property
            def std(self):
                return np.sqrt(self.M2 / (self.n - 1))

        # Usage
        

        theta_opt = [v.X + FVA_epsilon  for v in renzo.getVars() if "theta_" in v.VarName]
        tvars.ub = np.array(theta_opt).reshape(-1,1)
        target_vars = [v for v in renzo.getVars() if ("prot_" not in v.VarName) and ("theta_"not in v.VarName)]
        n_target_vars = len(target_vars)

        renzo.setParam("Threads", 4)
        dicti = {"Variable":[],"minFlux":[],"maxFlux":[]}
        stats_max = IterativeStats(n_vars=n_target_vars)
        stats_min = IterativeStats(n_vars=n_target_vars)
        for variable in range(n_target_vars):
            print(f"processing variable: {target_vars[variable].VarName}")
            renzo.setObjective(target_vars[variable],gp.GRB.MAXIMIZE)
            renzo.update()
            renzo.optimize()
            max_flux = renzo.ObjVal
            x = np.array([v.X for v in target_vars])
            stats_max.update(x)

            renzo.setObjective(target_vars[variable],gp.GRB.MINIMIZE)
            renzo.update()
            renzo.optimize()
            min_flux = renzo.ObjVal
            x = np.array([v.X for v in target_vars])
            stats_min.update(x)

            dicti["Variable"].append(target_vars[variable].VarName)
            dicti["maxFlux"].append(max_flux)
            dicti["minFlux"].append(min_flux)

        res_df = pd.DataFrame(dicti)

        res_df["meanFlux_Maximization"] = stats_max.mean[:res_df.shape[0]]
        res_df["stdFlux_Maximization"] = stats_max.std[:res_df.shape[0]]
        res_df["meanFlux_Minimization"] = stats_min.mean[:res_df.shape[0]]
        res_df["stdFlux_Minimization"] = stats_min.std[:res_df.shape[0]]

        return(res_df)





    def create_room_problem(self, enz_abundances, growth_rate, gamma, delta, epsilon):
        """ Creates a ROOM-like optimization problem which corrects turnover numbers, based on enzyme abundances and
            wild-type turnover numbers, such that the model can achieve a given growth rate.

        The main assumption is that, compared to a wild-type, only few turnover numbers change in a mutant-strain, but these
        changes are large.

        Since enzyme capacity constraint rely on both turnover numbers and enzyme abundances, we need to fix enzyme abundances,
        to avoid quadratic constraints.

        :param model: GEM object
        :param enz_abundances: dictionary of enzyme abundances. key - enzyme ID, value - abundance
        :param growth_rate: float, growth rate
        :param gamma: float, a factor describing the maximum change of turnover numbers.
                      The bounds (kcat_min, kcat_max) will be computed as (kcat' * (1/gamma), kcat' * gamma),
                      where kcat' is the wild-type turnover number.
        :param delta: float, relative change in computation of significance thresholds.
        :param epsilon: float, absolute change in computation of significance thresholds.

        :return: gurobipy.Model() object
        """
        model = self.gem
        S = deepcopy(model.S).toarray()
        enzymes = model.enzymes

        # remove total enzyme constraint, since enzyme amounts are fixed (sum_i mw_i*[E_i] = Ptot)
        prot_pool_idx = model.get_met_by_id("prot_pool")
        S = np.delete(S, prot_pool_idx, axis=0)

        n_mets, n_rxns = S.shape
        prot_idx = [model.get_met_by_id(f"{model.prot_pfx}{enz}") for enz in model.enzymes]
        met_idx = [i for i in range(n_mets) if i not in prot_idx]
        met_idx = met_idx[:-1]  # remove prot_pool

        # get kcat values and replace them with enzyme abundances in S
        kcat_dicti = {}
        for enz in enzymes:
            rxns = model.get_catalyzed_rxns(enz)
            rxn_idx = [model.get_rxn_by_id(r) for r in rxns]
            i = model.get_met_by_id(f"prot_{enz}")
            j = model.get_rxn_by_id(f"draw_prot_{enz}")
            kcats = [-1 / S[i, j] for j in rxn_idx]
            if len(np.unique(kcats)) > 1:
                raise ValueError(f"Multiple non-unique kcats for enzyme '{enz}' found")

            kcat_dicti[enz] = kcats[0]
            coeff = 0 if enz_abundances[enz] <= 1e-9 else -1 / (enz_abundances[enz] * 1.01)
            S[i, rxn_idx] = -1
            S[i, j] = enz_abundances[enz]

        ## FBA constraints
        # equality constraints
        Aeq = S[met_idx, :]
        beq = np.zeros(len(met_idx))
        # Aeq = S
        # beq = np.zeros(S.shape[0])

        # inequality constraints (enzyme capacity, but replace kcat with enzyme abundances)
        Aineq = -S[prot_idx, :]
        bineq = np.zeros(len(prot_idx))
        # Aineq = np.zeros((0, S.shape[1]))
        # bineq = np.zeros(0)

        ## ROOM CONSTRAINTS
        # Extend matrices for y-variables
        n_enz = len(enzymes)
        Aeq = np.hstack((Aeq, np.zeros((Aeq.shape[0], n_enz))))
        Aineq = np.hstack((Aineq, np.zeros((Aineq.shape[0], n_enz))))

        # compute bounds and significance thresholds for kcats
        kcats = np.array([kcat_dicti[enz] for enz in enzymes])
        kcat_min = kcats * (1 / gamma)
        kcat_max = kcats * gamma
        kcat_u = kcats + delta * kcats + epsilon
        kcat_l = kcats - delta * kcats - epsilon

        # create constraint matrices
        room_constraints_l = np.zeros((2 * n_enz, n_rxns))
        for k, enz in enumerate(enzymes):
            i = model.get_rxn_by_id(f"draw_prot_{enz}")
            room_constraints_l[k, i] = 1
            room_constraints_l[k + n_enz, i] = -1

        I = np.identity(n_enz)
        room_constraints_ru = I * -(kcat_max - kcat_u)
        room_constraints_rl = I * (kcat_min - kcat_l)

        room_constraints_r = np.vstack((room_constraints_ru, room_constraints_rl))
        room_constraints = np.hstack((room_constraints_l, room_constraints_r))
        room_b = np.hstack((kcat_u, -kcat_l))
        Aineq = np.vstack((Aineq, room_constraints))
        bineq = np.hstack((bineq, room_b))

        # Create arrays for variable initiation (bounds, names, variable type)
        var_names = model.rxns
        var_names_y = []
        lb = model.lb.flatten()
        ub = model.ub.flatten()
        for k, enz in enumerate(enzymes):
            rxn_idx = model.get_rxn_by_id(f"draw_prot_{enz}")
            lb[rxn_idx] = kcat_min[k]
            ub[rxn_idx] = kcat_max[k]
            var_names[rxn_idx] = f"kcat_{enz}"
            var_names_y.append(f"y_{enz}")

        # extend variable arrays for y-variables
        var_types = [GRB.CONTINUOUS] * n_rxns + [GRB.BINARY] * n_enz
        lb = np.hstack((lb, [0] * n_enz))
        ub = np.hstack((ub, [1] * n_enz))
        var_names = np.hstack((var_names, var_names_y))

        # fix growth rate (v_biomass >= growth_rate)
        bm_idx = model.get_rxn_by_id(model.biom_rxn)
        lb[bm_idx] = growth_rate

        # create optimization problem and add variables and constraints
        room = gp.Model()
        mvars = room.addMVar(len(var_names), lb=lb, ub=ub, vtype=var_types, name=var_names)
        room.addMConstr(Aeq, mvars, GRB.EQUAL, beq)
        room.addMConstr(Aineq, mvars, GRB.LESS_EQUAL, bineq)
        room.update()

        # add objective function
        y_vars = np.array([var for var in room.getVars() if var.VarName.startswith('y_')])
        lin_expr = gp.LinExpr(y_vars @ np.ones(len(y_vars)))
        room.setObjective(lin_expr, sense=GRB.MINIMIZE)
        room.update()

        self.room = room
    
    def create_room_problem_fixed(self, enz_abundances, growth_rate, gamma, delta, epsilon):
        """ Creates a ROOM-like optimization problem which corrects turnover numbers, based on enzyme abundances and
            wild-type turnover numbers, such that the model can achieve a given growth rate.

        The main assumption is that, compared to a wild-type, only few turnover numbers change in a mutant-strain, but these
        changes are large.

        Since enzyme capacity constraint rely on both turnover numbers and enzyme abundances, we need to fix enzyme abundances,
        to avoid quadratic constraints.

        :param model: GEM object
        :param enz_abundances: dictionary of enzyme abundances. key - enzyme ID, value - abundance
        :param growth_rate: float, growth rate
        :param gamma: float, a factor describing the maximum change of turnover numbers.
                      The bounds (kcat_min, kcat_max) will be computed as (kcat' * (1/gamma), kcat' * gamma),
                      where kcat' is the wild-type turnover number.
        :param delta: float, relative change in computation of significance thresholds.
        :param epsilon: float, absolute change in computation of significance thresholds.

        :return: gurobipy.Model() object
        """
        model = self.gem
        S = deepcopy(model.S).toarray()
        enzymes = model.enzymes

        # remove total enzyme constraint, since enzyme amounts are fixed (sum_i mw_i*[E_i] = Ptot)
        prot_pool_idx = model.get_met_by_id("prot_pool")
        S = np.delete(S, prot_pool_idx, axis=0)

        n_mets, n_rxns = S.shape
        prot_idx = [model.get_met_by_id(f"{model.prot_pfx}{enz}") for enz in model.enzymes]
        met_idx = [i for i in range(n_mets) if i not in prot_idx]
        met_idx = met_idx[:-1]  # remove prot_pool

        # get kcat values and replace them with enzyme abundances in S
        kcat_dicti = {}
        for enz in enzymes:
            rxns = model.get_catalyzed_rxns(enz)
            rxn_idx = [model.get_rxn_by_id(r) for r in rxns]
            i = model.get_met_by_id(f"prot_{enz}")
            j = model.get_rxn_by_id(f"draw_prot_{enz}")
            kcats = [-1 / S[i, j] for j in rxn_idx]
            if len(np.unique(kcats)) > 1:
                raise ValueError(f"Multiple non-unique kcats for enzyme '{enz}' found")

            kcat_dicti[enz] = kcats[0]
            coeff = 0 if enz_abundances[enz] <= 1e-9 else -1 / (enz_abundances[enz])
            S[i, rxn_idx] = 1
            S[i, j] = -enz_abundances[enz]

        ## FBA constraints
        # equality constraints
        Aeq = S[met_idx, :]
        beq = np.zeros(len(met_idx))
        # Aeq = S
        # beq = np.zeros(S.shape[0])

        # inequality constraints (enzyme capacity, but replace kcat with enzyme abundances)
        Aeq2 = S[prot_idx, :]
        #beq2= np.zeros(len(prot_idx))

        Aeq = np.vstack((Aeq, Aeq2))
        beq = np.zeros(Aeq.shape[0])
        # Aineq = np.zeros((0, S.shape[1]))
        # bineq = np.zeros(0)

        ## ROOM CONSTRAINTS
        # Extend matrices for y-variables
        n_enz = len(enzymes)
        Aeq = np.hstack((Aeq, np.zeros((Aeq.shape[0], n_enz))))
        #Aineq = np.hstack((Aineq, np.zeros((Aineq.shape[0], n_enz))))

        # compute bounds and significance thresholds for kcats
        kcats = np.array([kcat_dicti[enz] for enz in enzymes])
        kcat_min = kcats * (1 / gamma)
        kcat_max = kcats * gamma
        kcat_u = kcats + delta * kcats + epsilon
        kcat_l = kcats - delta * kcats - epsilon

        # create constraint matrices
        room_constraints_l = np.zeros((2 * n_enz, n_rxns))
        for k, enz in enumerate(enzymes):
            i = model.get_rxn_by_id(f"draw_prot_{enz}")
            room_constraints_l[k, i] = 1
            room_constraints_l[k + n_enz, i] = -1

        I = np.identity(n_enz)
        room_constraints_ru = I * -(kcat_max - kcat_u)
        room_constraints_rl = I * (kcat_min - kcat_l)

        room_constraints_r = np.vstack((room_constraints_ru, room_constraints_rl))
        room_constraints = np.hstack((room_constraints_l, room_constraints_r))
        room_b = np.hstack((kcat_u, -kcat_l))
        Aineq = room_constraints
        bineq = room_b

        # Create arrays for variable initiation (bounds, names, variable type)
        var_names = model.rxns
        var_names_y = []
        lb = model.lb.flatten()
        ub = model.ub.flatten()
        for k, enz in enumerate(enzymes):
            rxn_idx = model.get_rxn_by_id(f"draw_prot_{enz}")
            lb[rxn_idx] = kcat_min[k]
            ub[rxn_idx] = kcat_max[k]
            var_names[rxn_idx] = f"kcat_{enz}"
            var_names_y.append(f"y_{enz}")

        # extend variable arrays for y-variables
        var_types = [GRB.CONTINUOUS] * n_rxns + [GRB.BINARY] * n_enz
        lb = np.hstack((lb, [0] * n_enz))
        ub = np.hstack((ub, [1] * n_enz))
        var_names = np.hstack((var_names, var_names_y))

        # fix growth rate (v_biomass >= growth_rate)
        bm_idx = model.get_rxn_by_id(model.biom_rxn)
        lb[bm_idx] = growth_rate 

        # create optimization problem and add variables and constraints
        room = gp.Model()
        mvars = room.addMVar(len(var_names), lb=lb, ub=ub, vtype=var_types, name=var_names)
        room.addMConstr(Aeq, mvars, GRB.EQUAL, beq)
        room.addMConstr(Aineq, mvars, GRB.LESS_EQUAL, bineq)
        room.update()

        # add objective function
        y_vars = np.array([var for var in room.getVars() if var.VarName.startswith('y_')])
        lin_expr = gp.LinExpr(y_vars @ np.ones(len(y_vars)))
        room.setObjective(lin_expr, sense=GRB.MINIMIZE)
        room.update()

        self.room = room
    
    