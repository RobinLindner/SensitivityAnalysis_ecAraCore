import numpy as np
import pandas as pd
import gurobipy as gp
import scipy as sp
from gurobipy import GRB

from source import RESULT_DIR

class SensitivityAnalysis:

    def __init__(self, gem, inequality_idx=None):
        self.gem = gem.copy()
        self.prot_rxn_pfx = "draw_prot_"
        S = gem.S
        m, n = S.shape  
        # only Enzyme draw reactions
        self.enz_idx = [self.gem.get_rxn_by_id(f"{self.prot_rxn_pfx}{enz}") for enz in self.gem.enzymes]
        # protein pool exchange
        self.prot_pool_ex_idx = self.gem.get_rxn_by_id("prot_pool_exchange")
        # metabolic reactions
        self.rxn_idx = [i for i in range(n) if i not in (self.enz_idx or self.prot_pool_ex_idx)]
        # enzyme metabolites
        self.prot_idx = [self.gem.get_met_by_id(f"prot_{enz}") for enz in self.gem.enzymes]
        # protein pool metabolite
        self.prot_pool_id = self.gem.get_met_by_id("prot_pool")
        # metabolic metabolites
        self.met_idx = [i for i in range(m) if i not in (self.prot_idx or self.prot_pool_id)]
        if inequality_idx is not None:
            self.inequality_const = inequality_idx
            self.equality_const = np.setdiff1d(np.arange(m),inequality_idx)
        else:
            self.equality_const = np.arange(m)
            self.inequality_const = None
        self.primal = None
        self.dual = None
        self.enz_ub_set = False

    def GetTurnoverNumbersMode(self):
        """returns the kcat mode for each enzyme as a data frame"""
        d = {"Enzyme":[],"kcat":[]}
        for enz in self.gem.enzymes:
            idx = self.gem.get_met_by_id(f"prot_{enz}")
            S_d = self.gem.S[idx,self.rxn_idx]
            if(sp.sparse.issparse(S_d)):
                S_d = S_d.todense()
            S_d = np.asarray(S_d)
            idxs = np.where(S_d!=0)
            kcat = np.array(S_d[idxs])
            kcat_mode = sp.stats.mode(kcat)
            d["Enzyme"].append(enz)
            d["kcat"].append(kcat_mode.mode)
        return pd.DataFrame(d)

    def SolvePrimal(self):
        if(not self.enz_ub_set):
            print("Warning: Enzyme upper bounds not set. Setting to 1000.")
            self.SetUpperEnzymebounds(1000)
        
        primal_model = gp.Model("primal_lp")
        primal_model.setParam('NumericFocus', 3)  # not really needed, but removes numerical stability as the potential issue
        primal_model.Params.OptimalityTol = 1e-9
        primal_model.Params.FeasibilityTol = 1e-9
        primal_model.Params.MarkowitzTol = 0.999  # more stable pivots

        S = self.gem.S
        lb = self.gem.lb
        ub = self.gem.ub
        c = self.gem.c
        m, n = S.shape

        varnames = np.reshape(self.gem.rxns, lb.shape)
        # variables
        mvars = primal_model.addMVar(shape=lb.shape, lb=lb, ub=ub, obj=c,
                                        name=varnames, vtype=gp.GRB.CONTINUOUS) # unrestricted

        mvars = mvars.reshape(-1)

        # get protein indices
        #prot_idx = [self.gem.get_met_by_id(f"prot_{enz}") for enz in self.gem.enzymes]
        #met_idx = [i for i in range(m) if i not in prot_idx]
        # equality constraints
        Aeq = self.gem.S[self.equality_const,:]
        beq = self.gem.b[self.equality_const].reshape(-1)
        primal_model.addMConstr(A=Aeq, x=mvars, b=beq, sense=gp.GRB.EQUAL, name=self.gem.mets[self.equality_const])
        # equality constraints (enzyme abundance)
        if self.inequality_const is not None:
            Aineq = self.gem.S[self.inequality_const, :]
            bineq = self.gem.b[self.inequality_const].reshape(-1)
            primal_model.addMConstr(A=Aineq, x=mvars, b=bineq, sense=gp.GRB.LESS_EQUAL, name=self.gem.mets[self.inequality_const])

        primal_model.update()
        
        primal_model.setObjective(gp.quicksum(c[i] * mvars[i] for i in range(len(c))),GRB.MAXIMIZE)

        primal_model.update()
        # Solve
        primal_model.optimize()
        self.primal = primal_model
        # Get the primal solution
        if primal_model.status == GRB.OPTIMAL:
            self.primal_flux_sol = np.array([mvars[j].X for j in range(n)])
            self.primal_obj_val = primal_model.ObjVal
            print("Optimal dual objective:", self.primal_obj_val)
        else:
            print("Model not optimal. Status:", primal_model.status)

    def SolveDual(self):
        if(not self.enz_ub_set):
            print("Warning: Enzyme upper bounds not set. Setting to 1000.")
            self.SetUpperEnzymebounds(1000)
            
        dual_model = gp.Model("dual_lp")
        dual_model.setParam('NumericFocus', 3)  # not really needed, but removes numerical stability as the potential issue
        dual_model.Params.OptimalityTol = 1e-9
        dual_model.Params.FeasibilityTol = 1e-9
        dual_model.Params.MarkowitzTol = 0.999  # more stable pivots

        
        S = self.gem.S

        # number of reactions and enzymes
        n_rxn = len(self.rxn_idx)
        n_enz = len(self.enz_idx)

        # number of metabolites
        n_met = len(self.met_idx)
        n_prot = len(self.prot_idx)

        # Dual variables
        varnames = np.concatenate(([f"lambda_{i}" for i in range(n_met)],
                                [f"lambda_neg_{i}" for i in range(n_met)],
                                [f"Xi_{i}" for i in range(n_prot)],
                                [f"Xi_neg{i}" for i in range(n_prot)],
                                [f"mu_max_{i}" for i in range(n_rxn)],
                                [f"mu_min_{i}" for i in range(n_rxn)],
                                [f"epsilon_max_{i}" for i in range(n_enz)],
                                [f"epsilon_min_{i}" for i in range(n_enz)],
                                [f"epsilon_tot"]),0)

        mvars = dual_model.addMVar(shape=varnames.shape, lb=0, ub=np.inf,
                                        name=varnames, vtype=gp.GRB.CONTINUOUS) # unrestricted

        mvars = mvars.reshape(-1)

        m, n = S.shape

        enz_lb = self.gem.lb[self.enz_idx].reshape(-1,)
        enz_ub = self.gem.ub[self.enz_idx].reshape(-1,)
        
        ptot = self.gem.ub[self.prot_pool_ex_idx] # total proteome pool size

        rxn_lb = self.gem.lb[self.rxn_idx].reshape(-1,)
        rxn_ub = self.gem.ub[self.rxn_idx].reshape(-1,)

        c_rxn = self.gem.c[self.rxn_idx]
        c_enz = self.gem.c[self.enz_idx]

        c = np.concatenate([c_rxn,c_enz])

        b_met = self.gem.b[self.met_idx].reshape(-1,)
        b_prot = self.gem.b[self.prot_idx].reshape(-1,)

        obj_vec = np.concatenate((b_met,
                                b_met*-1,
                                b_prot,
                                b_prot*-1,
                                rxn_ub,
                                -rxn_lb,
                                enz_ub,
                                -enz_lb,
                                np.reshape(ptot,(1,))))

        # Set the objective function
        # minimize: 0 * lambda + 0 * lambda_neg + 0 * Xi + 0 * Xi_neg + mu_max * v + mu_min * -v + epsilon_max * e + epsilon_min * -e + epsilon_tot
        # which is equivalent to minimizing the sum of the upper bounds on the reactions and enzymes
        dual_model.setObjective(gp.quicksum(obj_vec[i] * mvars[i] for i in range(len(obj_vec))),GRB.MINIMIZE)

        
        # A^T = [ S^T, -S^T , 1/Kcat, -1/kcat,  1,  -1, 0, 0, 0;
        #         0  , 0    , -1    ,1       ,  0,   0, 1, -1, MW]
        if(sp.sparse.issparse(S)): 
            St = np.transpose(S).todense()
        else:
            St = np.transpose(S)

        MW_vec = St[:,self.prot_pool_id.reshape(-1)]

        Aineq_top = np.concatenate([St[np.ix_(self.rxn_idx,self.met_idx)],                            # Sv≤0 
                                            -1 * St[np.ix_(self.rxn_idx,self.met_idx)],                 # -Sv≤0
                                            -1 * St[np.ix_(self.rxn_idx,self.prot_idx)],               # v/kcat - e ≤0
                                            St[np.ix_(self.rxn_idx,self.prot_idx)],                   # - v/kcat + e ≤ 0
                                            np.eye(n_rxn),                                  # v ≤ vmax
                                            -1 * np.eye(n_rxn),                               # -v ≤ -vmin
                                            np.zeros((n_rxn,n_enz)),                        # e ≤ emax
                                            np.zeros((n_rxn,n_enz)),                        # -e ≤ -emin
                                            np.zeros((n_rxn,1))],1)                         # sum(e*MW) ≤ etot
        Aineq_bot = np.concatenate([np.zeros((n_enz,n_met)),                                # [repeat]
                                            np.zeros((n_enz,n_met)),
                                            -1 * St[np.ix_(self.enz_idx,self.prot_idx)],
                                            St[np.ix_(self.enz_idx,self.prot_idx)],
                                            np.zeros((n_enz,n_rxn)),
                                            np.zeros((n_enz,n_rxn)),
                                            np.eye(n_enz),
                                            -1 * np.eye(n_enz),
                                            -1 * MW_vec[self.enz_idx]],1)
        Aineq = np.concatenate([Aineq_top,Aineq_bot],0)

        #print("before constraints")
        #print(Aineq.shape)
        #print(mvars.shape)


        c=c.reshape(-1)
        #print(c.shape)

        dual_model.addMConstr(A=Aineq, x=mvars, b=c, sense=gp.GRB.GREATER_EQUAL)#, name=np.concatenate(model.rxns[rxn_idx],model.rxns[enz_idx]))
        print("=== GUROBI UPDATE===\n\n")
        dual_model.update()

        print("=== GUROBI Optimize===\n\n")
        dual_model.optimize()

        k=0
        self.dual = dual_model
        # Get the dual solution
        if dual_model.status == GRB.OPTIMAL:
            self.lambda_sol = np.array([mvars[i].X for i in range(k,n_met)])
            k=n_met
            self.lambda2_sol = np.array([mvars[i].X for i in range(k,k+n_met)])
            k=k+n_met
            self.Xi_sol = np.array([mvars[i].X for i in range(k,k+n_prot)])
            k = k+n_prot
            self.Xi2_sol = np.array([mvars[i].X for i in range(k,k+n_prot)])
            k = k+n_prot
            self.mu_max_sol = np.array([mvars[i].X for i in range(k,k+n_rxn)])
            k = k+n_rxn
            self.mu_min_sol = np.array([mvars[i].X for i in range(k,k+n_rxn)])
            k = k+n_rxn
            self.epsilon_max_sol = np.array([mvars[i].X for i in range(k,k+n_enz)])
            k = k+n_enz
            self.epsilon_min_sol = np.array([mvars[i].X for i in range(k,k+n_enz)])
            k = k+n_enz
            self.epsilon_tot_sol = np.array([mvars[k].X])
            self.dual_obj_val = dual_model.ObjVal
            print("==== GUROBI status ====\n")
            print("Optimal dual objective:", self.dual_obj_val)
            print("====  ====\n")
        else:
            print("==== GUROBI status ====\n")
            print("Model not optimal, Status:", dual_model.status)
            print("====  ====\n")

    def solvePFBA(self,alpha=0.95,variable_selection = 0):
        if self.primal is None:
            self.SolvePrimal()
        self.pfba = self.primal.copy()
        # get objective value from solved FBA
        z_star = self.primal.ObjVal
        obj_fun = self.primal.Obj
        sense = self.primal.sense

        # add FBA objective as constraint
        vars = self.pfba.getVars()
        expr = gp.LinExpr(obj_fun, vars)
        if sense == 1:
            self.pfba.addLConstr(expr, GRB.LESS_EQUAL, z_star*(2-alpha), 'prev_objective')
        else:
            self.pfba.addLConstr(expr, GRB.GREATER_EQUAL, z_star*alpha, 'prev_objective')

        # add auxiliary absolute value variables
        if variable_selection == 1:
            vars = self.pfba.getVars()
        elif variable_selection == 0:
            vars = [v for v in self.pfba.getVars() if not (("draw_prot_" in v.VarName) | ("prot_pool_" in v.VarName))]
        else:
            vars = [v for v in self.pfba.getVars() if "draw_prot_" in v.VarName]


        n_vars = len(vars)
        abs_vars = self.pfba.addVars(n_vars, lb=0, name="abs_aux")
        self.pfba.update()

        for i, var in enumerate(vars):
            self.pfba.addConstr(abs_vars[i] >= var,  f"abs_pos_{i}")
            self.pfba.addConstr(abs_vars[i] >= -var, f"abs_neg_{i}")

        print(f"Optimal baseline biomass flux: {z_star}")
        print(f"pFBA minimal biomass bound: {z_star*alpha}")
        print(f"Fluxes being minimized: {n_vars}")

        # minimize sum of absolute fluxes using aux variables
        expr = gp.LinExpr(np.ones((n_vars,)), [abs_vars[i] for i in range(n_vars)])
        self.pfba.setObjective(expr, GRB.MINIMIZE)
        self.pfba.optimize()

        fluxes = pd.Series({v.VarName: v.X for v in self.pfba.getVars() if "abs_aux" not in v.VarName})

        return fluxes



    def GetShadowPrices(self):
        dicti = {"Element": [],"Constraint":[],"Shadow price":[]}
        n_met = len(self.met_idx)
        n_prot = len(self.prot_idx)
        n_rxn = len(self.rxn_idx)
        n_enz = len(self.enz_idx)
        print(n_met)
        print(n_prot)
        print(n_rxn)
        print(n_enz)
        for i in range(n_met):
            dicti["Element"].append(self.gem.mets[self.met_idx[i]])
            dicti["Constraint"].append("Lower steady state")
            dicti["Shadow price"].append(self.lambda_sol[i])
        for i in range(n_met):
            dicti["Element"].append(self.gem.mets[self.met_idx[i]])
            dicti["Constraint"].append("Upper steady state")
            dicti["Shadow price"].append(self.lambda2_sol[i])
        for i in range(n_prot):
            dicti["Element"].append(self.gem.mets[self.prot_idx[i]])
            dicti["Constraint"].append("Upper enzyme rate limiting")
            dicti["Shadow price"].append(self.Xi_sol[i])
        for i in range(n_prot):
            dicti["Element"].append(self.gem.mets[self.prot_idx[i]])
            dicti["Constraint"].append("Lower enzyme rate limiting")
            dicti["Shadow price"].append(self.Xi2_sol[i])
        for i in range(n_rxn):
            dicti["Element"].append(self.gem.rxns[self.rxn_idx[i]])
            dicti["Constraint"].append("Upper flux bound")
            dicti["Shadow price"].append(self.mu_max_sol[i])
        for i in range(n_rxn):
            dicti["Element"].append(self.gem.rxns[self.rxn_idx[i]])
            dicti["Constraint"].append("Lower flux bound")
            dicti["Shadow price"].append(self.mu_min_sol[i])
        for i in range(n_enz):
            dicti["Element"].append(self.gem.rxns[self.enz_idx[i]])
            dicti["Constraint"].append("Upper enzyme bound")
            dicti["Shadow price"].append(self.epsilon_max_sol[i])
        for i in range(n_enz):
            dicti["Element"].append(self.gem.rxns[self.enz_idx[i]])
            dicti["Constraint"].append("Lower enzyme bound")
            dicti["Shadow price"].append(self.epsilon_min_sol[i])
        dicti["Element"].append("Ptot")
        dicti["Constraint"].append(f"Total proteome capacity")
        dicti["Shadow price"].append(self.epsilon_tot_sol[0])     
        df = pd.DataFrame(dicti)   
        df["Element"] = df["Element"].astype(str)
        df["Constraint"] = df["Constraint"].astype(str)
        return df  

    def ResetProblems(self):
        self.primal = None
        self.dual = None

    def ReturnProblems(self):
        return([self.primal,self.dual])

    def ComputeSensitivityCoefficients(self,outpath = None):
        if(self.primal is None):
            print("Recomputing primal.")
            self.SolvePrimal()
        if(self.dual is None):
            print("Recomputing dual.")
            self.SolveDual()
        
        if(self.primal.Status != GRB.OPTIMAL or self.dual.Status != GRB.OPTIMAL):
            print("Error: Primal or dual problem not optimal. Cannot compute sensitivity coefficients.")
            df = pd.DataFrame({'Type': ["Infeasible"],'SensitivityCoefficient': ["Infeasible"], 'ModelID':["Infeasible"]})
            return df

        dicti = {'Type': [],'SensitivityCoefficient': [], 'ModelID':[]}
        vs = self.primal_flux_sol[self.rxn_idx]
        rxns = self.gem.rxns[self.rxn_idx]
        assert len(self.mu_max_sol) == len(self.mu_min_sol) and len(self.mu_max_sol) == len(vs), \
            f'ALARM: {len(self.mu_max_sol)}, {len(self.mu_min_sol)}, {len(self.primal_sol)}'
        
        
        
        fCSC = np.zeros(len(self.mu_max_sol))
        for i in range(len(vs)):
            dicti['Type'].append("FluxCapacity")
            dicti['SensitivityCoefficient'].append( vs[i]/self.dual_obj_val * (self.mu_max_sol[i] - self.mu_min_sol[i]))
            dicti['ModelID'].append(rxns[i])
            

        # enzyme CSCs
        es = self.primal_flux_sol[self.enz_idx]
        enzymes = self.gem.rxns[self.enz_idx]
        assert len(self.epsilon_max_sol) == len(self.epsilon_min_sol) and len(self.epsilon_max_sol) == len(es), \
            f'ALARM: {len(self.epsilon_max_sol)}, {len(self.epsilon_min_sol)}, {len(es)}'
        eCSC = np.zeros(len(self.mu_max_sol))
        for i in range(len(es)):
            eCSC[i] = es[i]/self.dual_obj_val * (self.epsilon_max_sol[i] - self.epsilon_min_sol[i])
            dicti['Type'].append("EnzymeCapacity")
            dicti['SensitivityCoefficient'].append(es[i]/self.dual_obj_val * (self.epsilon_max_sol[i] - self.epsilon_min_sol[i]))
            dicti['ModelID'].append(enzymes[i])
        
        # proteome CSC
        # WENN DU ETOT EXPLIZIT IMPLEMENTIERT HAST IST DAS JETZT VLLT ANDERS
        
        dicti['Type'].append("ProteomeCapacity")
        dicti['SensitivityCoefficient'].append(((self.gem.ub[self.prot_pool_ex_idx] / self.dual_obj_val) * self.epsilon_tot_sol[0]).flatten()[0])
        dicti['ModelID'].append("prot_pool_exchange")
        
        # ESC
        assert len(es) == len(self.Xi_sol), \
            f'ALARM: {len(self.epsilon_max_sol)}, {len(self.epsilon_min_sol)}, {len(es)}'
        for i in range(len(es)):
            dicti['Type'].append("ESC")
            dicti['SensitivityCoefficient'].append(es[i] / self.dual_obj_val * (self.Xi_sol[i]-self.Xi2_sol[i]))
            dicti['ModelID'].append(enzymes[i])


        sol_df = pd.DataFrame(dicti)
        if outpath is not None:
            if not outpath.endswith('.tsv'):
                outpath += '.tsv'
            sol_df.to_csv(RESULT_DIR/ outpath,sep="\t", index=False)
        
        return sol_df   

    def ComputeSensitivityCoefficients_alt(self,outpath = None):
            if(self.primal is None):
                print("Recomputing primal.")
                self.SolvePrimal()

            if(self.primal.Status != GRB.OPTIMAL):
                print("Error: Primal problem not optimal. Cannot compute sensitivity coefficients.")
                df = pd.DataFrame({'Type': ["Infeasible"],'SensitivityCoefficient': ["Infeasible"], 'ModelID':["Infeasible"]})
                return df
            
            dicti = {'Type': [],'SensitivityCoefficient': [], 'ModelID':[]}
            vars = np.array([v for v in self.primal.getVars()])
            constrs = np.array([c for c in self.primal.getConstrs()])
            vs = self.primal_flux_sol[self.rxn_idx]
            rxns = self.gem.rxns[self.rxn_idx]

            fvars = vars[self.rxn_idx]
            '''
            assert len(self.mu_max_sol) == len(self.mu_min_sol) and len(self.mu_max_sol) == len(vs), \
                f'ALARM: {len(self.mu_max_sol)}, {len(self.mu_min_sol)}, {len(self.primal_sol)}'
            '''
            
            
            #fCSC = np.zeros(len(self.mu_max_sol))
            for i in range(len(vs)):
                dicti['Type'].append("FluxCapacity")
                dicti['SensitivityCoefficient'].append(vs[i] / self.primal_obj_val * (fvars[i].RC))
                dicti['ModelID'].append(rxns[i])
                
    
            # enzyme CSCs
            es = self.primal_flux_sol[self.enz_idx]
            enzymes = self.gem.rxns[self.enz_idx]
            evars = vars[self.enz_idx]
            '''
            assert len(self.epsilon_max_sol) == len(self.epsilon_min_sol) and len(self.epsilon_max_sol) == len(es), \
                f'ALARM: {len(self.epsilon_max_sol)}, {len(self.epsilon_min_sol)}, {len(es)}'
            '''
            #eCSC = np.zeros(len(self.mu_max_sol))
            for i in range(len(es)):
                #eCSC[i] = es[i]/self.dual_obj_val * (self.epsilon_max_sol[i] - self.epsilon_min_sol[i])
                dicti['Type'].append("EnzymeCapacity")
                dicti['SensitivityCoefficient'].append(es[i]/self.primal_obj_val * (evars[i].RC))
                dicti['ModelID'].append(enzymes[i])
            
            # proteome CSC
            # WENN DU ETOT EXPLIZIT IMPLEMENTIERT HAST IST DAS JETZT VLLT ANDERS
            pcvar= vars[self.prot_pool_ex_idx]
            dicti['Type'].append("ProteomeCapacity")
            dicti['SensitivityCoefficient'].append(((self.gem.ub[self.prot_pool_ex_idx] / self.primal_obj_val) * pcvar.RC).flatten()[0])
            dicti['ModelID'].append("prot_pool_exchange")
            
            # ESC
            '''
            assert len(es) == len(self.Xi_sol), \
                f'ALARM: {len(self.epsilon_max_sol)}, {len(self.epsilon_min_sol)}, {len(es)}'
            '''
            econsts = constrs[self.prot_idx]
            for i in range(len(es)):
                dicti['Type'].append("ESC")
                dicti['SensitivityCoefficient'].append(es[i] / self.primal_obj_val * (econsts[i].Pi * (-1)))
                dicti['ModelID'].append(enzymes[i])
            
    
            sol_df = pd.DataFrame(dicti)
            if outpath is not None:
                if not outpath.endswith('.tsv'):
                    outpath += '.tsv'
                sol_df.to_csv(RESULT_DIR/ outpath,sep="\t", index=False)
            
            return sol_df

    