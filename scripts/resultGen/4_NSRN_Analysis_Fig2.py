import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, io 
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
import scipy.stats as sp
from statsmodels.stats import multitest
root_dir = Path(__file__).resolve().parents[2]

sys.path.append(str(root_dir))


from source import FIG_DIR, SUPP_FIG_DIR, ESC_DATA_WIDE, MODEL_ENZYME_2_PMET, MODEL_ENZYME_2_SUBSYSTEM, CLUST_RES_DIR


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



## Output paths
fig_out_path = FIG_DIR / "enzyme_sensitivity_trajectories_with_dendro.png"

supp_fig_out_path =  SUPP_FIG_DIR / "subsystem_cluster_proportions.png"

clust_out_path = CLUST_RES_DIR / "enzyme_clustering.csv"

clust_enrichment_out = CLUST_RES_DIR / "Cluster_enrichment.csv"

clust_subs_prop_out = CLUST_RES_DIR / "Cluster_subsystem_proportions.csv"

SHOW_FIGURES = False

def main():
    print("== CLUSTERING ==")

    # Load ESC data
    SC_wide = pd.read_csv(ESC_DATA_WIDE,index_col=0)

    # Min-max scale data (normalization)
    SC_norm = SC_wide.sub(SC_wide.min(axis=1),axis=0).div(SC_wide.max(axis=1)-SC_wide.min(axis=1),axis=0)
    full_df = SC_norm.dropna()
    print(f"Dropping {SC_norm.shape[0]-full_df.shape[0]} rows because of NA filtering.")
    print()
    # Transform to numpy array and add averages to pandas data frame
    X=full_df.to_numpy()
    averages = SC_wide.mean(axis=1)
    full_df.loc[:,"Average"] = averages[averages!=0]

    # Compute inertia of k means clustering for 1 to 11 clusters
    inertia =[]
    for i in np.arange(1,11):
        mod = KMeans(n_clusters=i,random_state=5)
        res = mod.fit_predict(X)
        inertia.append(mod.inertia_)
    plt.plot(np.arange(1,11),inertia)
    plt.title("Inertia by number of clusters.")
    plt.show(block=SHOW_FIGURES)
    mod = KMeans(n_clusters=4,random_state=5)
    clust_assignment = mod.fit_predict(X)
    sil_score = silhouette_score(X,clust_assignment)
    dist_mat = pairwise_distances(mod.cluster_centers_)
    print(f"The average silhouette score is {sil_score}.")
    print()

    # Dimensionality reduction - PCA
    pca = PCA()
    X_reduced = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_
    print(f"The first three prinical components explain {np.round(sum(explained_variance[0:3])*100,2)}% of the total variance.")
    print()

    # Dimensionality reduction - tSNE
    tsne = TSNE()
    X_embedded = tsne.fit_transform(X)
    sns.scatterplot(x=X_embedded[:,0],y=X_embedded[:,1],hue=res[1]).set_title("tSNE embedding")
    plt.show(block=SHOW_FIGURES)
    data = pd.Series(explained_variance,index=np.arange(1,len(explained_variance)+1)).reset_index()
    data.columns= ["Principal components","Explained variance"]
    sns.barplot(data,x="Principal components",y="Explained variance").set_title("PCA - explained variance by PC")
    plt.show(block=SHOW_FIGURES)
    averages = full_df["Average"]

    #res = k_means(X,n_clusters=4,random_state=5)
    #pca = PCA(n_components=3)
    #X_reduced = pca.fit_transform(X)



    # Cluster colors
    colormap = {3:"#d31f11",
                2:"#f47a00",
                0:"#A4C3B2",
                1:"#007191"}

    # Range of point sizes
    pointsMin = 0
    pointsMax = 200

    power_transform = [a**(1/2) for a in averages]
    pointsizes = [(a) * (pointsMax-pointsMin)+pointsMin + 3 for a in power_transform]

    # Figure layout
    fig = plt.figure(layout= "constrained",figsize=(12,6))
    fig.get_layout_engine().set(wspace=0.05) 
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.1, 1.5, 0.4]) # 4 rows, 2 columns

    # Left column - scatter plots
    gs_left = gs[0].subgridspec(2,1)
    ax1 = fig.add_subplot(gs_left[0])
    ax2 = fig.add_subplot(gs_left[1],sharex=ax1)

    # Center column: 4 line plots, each taking 1 row
    gs_cent= gs[2].subgridspec(4,1)
    ax3 = fig.add_subplot(gs_cent[0])
    ax4 = fig.add_subplot(gs_cent[1],sharex=ax3)
    ax5 = fig.add_subplot(gs_cent[2],sharex=ax3)
    ax6 = fig.add_subplot(gs_cent[3],sharex=ax3)

    # right column: 1 dendrogram for cluster distance
    #gs_right= gs[4].subgridspec(1,1)
    ax7 = fig.add_subplot(gs[3])



    # Dendrogramm first to extract plot order 
    print(dist_mat)
    condensed = squareform(dist_mat)   # converts n×n matrix to condensed 1D form scipy expects
    Z = linkage(condensed, method="average")   # or "complete", "single", "ward", etc.
    res = dendrogram(Z, labels=np.arange(4), ax=ax7, orientation="right")
    ax7.set_ylabel("")
    ax7.set_yticks([])
    ax7.set_xlabel("Centroid distance",
                fontsize=13)
    cluster_order = res["ivl"]
    


    ax=[ax1,ax2,ax3,ax4,ax5,ax6]
    labelmap = {0:"Spurious",
                1:"Low",
                2:"Continuous",
                3:"High"}
    '''
    

    group_order = [
         "Continuous",
         "High",
         "Low",
         "Spurious",
    ]
    '''
    group_reorder = {i:cluster_order[i] for i in np.arange(4)} 

    group_assignment = np.zeros_like(clust_assignment)
    for i in np.arange(0,4):
        group_assignment[clust_assignment==i] = group_reorder[i] 

    clust_assignment = group_assignment
    
    #group_assignment = res
    #print(group_assignment)
    outline = np.full(len(averages), 'none', dtype=object)
    a_np = np.array(averages)
    for clust in np.unique(clust_assignment):
        mask = clust_assignment == clust
        group_max_idx = np.where(mask)[0][np.argmax(a_np[mask])]
        outline[group_max_idx] = "black"

    sc = ax[0].scatter(X_reduced[:,0],X_reduced[:,1],pointsizes,c=[colormap[c] for c in clust_assignment],edgecolors=outline)

    ax[1].scatter(X_reduced[:,0],X_reduced[:,2],pointsizes,c=[colormap[c] for c in clust_assignment],edgecolors=outline)

    rel_idx = np.where(outline != "none")[0]
    ax[0].scatter(X_reduced[rel_idx,0],X_reduced[rel_idx,1],np.array(pointsizes)[rel_idx],c=[colormap[c] for c in clust_assignment[rel_idx]],edgecolors=outline[rel_idx])
    ax[1].scatter(X_reduced[rel_idx,0],X_reduced[rel_idx,2],np.array(pointsizes)[rel_idx],c=[colormap[c] for c in clust_assignment[rel_idx]],edgecolors=outline[rel_idx])

    max_labels = full_df.index[rel_idx]
    enzyme_name_map = {
        "F4IHR4" : "$CA$-$II$",
        "O03042" : "$rbcL$",
        "P56775" : "$petG$",
        "Q9ZUC2" : "$\\beta$-$CA$-$III$",
        "P10798" : "$RBCS$-$3B$"
    } 

    for i in np.arange(len(rel_idx)):
        x_offset = 0.1
        y_offset = 0.1
        if(enzyme_name_map[full_df.index[rel_idx[i]]] in  ["$\\beta$-$CA$-$III$","$RBCS$-$3B$","$CA$-$II$"]):
            ha = "right"
            va = "bottom"
            rot=-30
        else:
            ha = "left"
            va = "top"
            rot=-30
        ax[0].text(x=X_reduced[int(rel_idx[i]),0] + x_offset,
                y=X_reduced[int(rel_idx[i]),1] + y_offset,
                s=enzyme_name_map[full_df.index[rel_idx[i]]],
                rotation = rot,
                weight= "bold",
                horizontalalignment = ha,
                verticalalignment = va)

    ax[0].set_xlabel("PC1")
    ax[1].set_xlabel("PC1")

    ax[0].set_ylabel("PC2")
    ax[1].set_ylabel("PC3")

    handles, labels = [], []
    cluster_avgs = []
    ## Creates lineplots for right side of figure
    for cluster in np.arange(0,4):
        print(labelmap[cluster])
        ## Extract data for enzymes of corresponding cluster
        subset = full_df[clust_assignment==cluster].rename_axis("Enzyme").reset_index()


        # Turn to long format
        plot_df = pd.melt(subset,id_vars=["Enzyme","Average"],value_vars=np.arange(10,41).astype(str),value_name="Normalized Sensitivity",var_name="Temperature")
        plot_df["Temperature"] = plot_df["Temperature"].astype(int)
        # Get data of largest sensitivity enzyme in cluster
        max_df = plot_df[plot_df["Average"] == np.max(plot_df["Average"])]
        # Get data for largest 90th quantile of sensitivity in cluster 
        q_90_df = plot_df[plot_df["Average"] >= np.quantile(subset["Average"],0.9)]

        view_df_1 = plot_df.drop(columns="Average").pivot(index="Enzyme",columns="Temperature",values="Normalized Sensitivity").mean(axis=0)
        view_df_2 = q_90_df.drop(columns="Average").pivot(index="Enzyme",columns="Temperature",values="Normalized Sensitivity").mean(axis=0)
        view_df_3 = max_df.drop(columns="Average").pivot(index="Enzyme",columns="Temperature",values="Normalized Sensitivity").mean(axis=0)
        print("Average normalized senstivity")
        print("of all enzymes in cluster:")
        print(np.round(view_df_1.loc[np.arange(10,41,2)].to_frame().T,2))
        print("of enzymes with highest 10\% average ESC in cluster:")
        print(np.round(view_df_2.loc[np.arange(10,41,2)].to_frame().T,2))
        print("of enzyme with highest average ESC in cluster:")
        print(np.round(view_df_3.loc[np.arange(10,41,2)].to_frame().T,2))
        print()
        # MAX ENZYME - dashed line
        sns.lineplot(max_df,x="Temperature",y="Normalized Sensitivity",
                    alpha=1,
                    ax=ax[cluster+2],
                    legend=None,
                    label = enzyme_name_map[max_df["Enzyme"].values[0]],
                    errorbar=('ci', 99),
                    color=colormap[cluster],
                    linestyle = "dashed"
                    )
        
        # TOP 90th quartile - gray dotted line
        sns.lineplot(q_90_df,x="Temperature",y="Normalized Sensitivity",
                    alpha=.6,
                    ax=ax[cluster+2],
                    legend=None,
                    label = "90th quantile",
                    errorbar=('ci', 99),
                    color="Gray",
                    linestyle = "dotted"
                    )
        
        # All enzymes - colored solid line
        sns.lineplot(plot_df,x="Temperature",y="Normalized Sensitivity",
                    alpha=1,
                    ax=ax[cluster+2],
                    legend=None,
                    label = labelmap[cluster],
                    errorbar=('ci', 99),
                    color=colormap[cluster],
                    )
        
        h, l = ax[cluster+2].get_legend_handles_labels()
        ax[cluster+2].text(x=10,y=0.8,s = f"$n: {len(np.unique(subset["Enzyme"]))}$")
        ax[cluster+2].set_ybound(0,1)
        handles.extend([h[i] for i in [2,0]])
        labels.extend([l[i] for i in [2,0]])

        cluster_avg=plot_df.loc[:,["Temperature","Normalized Sensitivity"]].groupby("Temperature").mean()
        cluster_avgs.append(cluster_avg)
        '''
        if(i==3):
            handles.extend([h[1]])
            labels.extend([l[1]])
        '''
        #print(f"The {i+1} group contains: \n{subset["Enzyme"].tolist()}")
        
    plt.setp([ax1.get_xticklabels()], visible=False)
    ax1.set_xlabel("")
    ax1.set_ylabel(f"PC2 ({np.round(explained_variance[1]*100,1)}%)",fontsize=13)
    ax2.set_xlabel(f"PC1 ({np.round(explained_variance[0]*100,1)}%)",fontsize=13)
    ax2.set_ylabel(f"PC3 ({np.round(explained_variance[2]*100,1)}%)",fontsize=13)
    plt.setp([ax3.get_xticklabels(), ax4.get_xticklabels(), ax5.get_xticklabels()], visible=False)
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    #ax3.set_title("Limiting at high temperature")
    ax4.set_xlabel("")
    ax4.set_ylabel(" ")
    #ax4.set_title("Limiting at optimal temperature")
    ax5.set_xlabel("")
    ax5.set_ylabel("")
    #ax5.set_title("Limiting at low temperature")
    ax6.set_ylabel("")
    ax6.set_xlabel("Temperature [°C]",
                fontsize=13)
    #ax6.set_title("Limiting at non-optimal temperature")
    fig.text(0.35, 0.5, 'Normalized Sensitivity', 
            va='center', rotation=90,fontsize=13)
    fig.text(0,1, "A", fontweight="bold", 
            va='center',fontsize=14)
    fig.text(0.35,1, "B", fontweight="bold", 
            va='center',fontsize=14)
    '''
    fig.text(0.85,1, "C", fontweight="bold", 
                va='center',fontsize=14)
    '''
    

    h,l = sc.legend_elements(prop="sizes",alpha = 0.6,num=[1e-3,1e-2,1e-1], func=lambda s: ((s-3)/pointsMax)**2)

    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.75, -0.05))

    fig.legend(h,l,loc="upper center",title="Average sensitivity",ncol=3, bbox_to_anchor=(0.25, -0.05))
    
    fig.savefig(fig_out_path, dpi=300, bbox_inches='tight')
    plt.show(block=SHOW_FIGURES)
    print(f"Figure saved at:")
    print(f"{fig_out_path}")
    print()

    print("Average standard deviation of ESC within clusters")
    for i in np.arange(4):
        Group1_df = full_df[group_assignment==i].rename_axis("Enzyme").reset_index()
        print(f"Avg std g1:{Group1_df.iloc[:,1:].std(axis=0).iloc[:-1].mean()}")

    print()

    print("== ISOENZYME PROPORTIONS ==")

    isoenzyme_map = pd.read_csv(MODEL_ENZYME_2_PMET,index_col=0)
    subsystem_map = pd.read_csv(MODEL_ENZYME_2_SUBSYSTEM,index_col=0)

    dicti = {"UniProtID":[],
             "Name":[],
             "Cluster":[],
             "Cluster_num":[],
             "Subsystem":[],
             "Isoenzyme":[],
             }

    groups = {0:"Spurious",
            1:"Low",
            2:"Continuous",
            3:"High"}
    
    groups_inv = {v: k for k, v in groups.items()}
    
    cluster_lookup = pd.DataFrame({"Enzymes":full_df.index,
                                    "Group" : pd.Series(group_assignment).map(groups)}).set_index("Enzymes").squeeze()


    name_lookup = subsystem_map.loc[:,["Enzyme","Name"]].drop_duplicates().set_index("Enzyme").squeeze().to_dict()
    #reaction_lookup = isoenzyme_map.loc[:,["Enzymes","Isorxns"]].drop_duplicates().set_index("Enzymes").squeeze().to_dict()
    subsystem_lookup = subsystem_map.loc[:,["Enzyme","Subsystem"]].groupby("Enzyme").agg(lambda x : "; ".join(x.astype(str))).squeeze()
    isoenzyme_lookup = isoenzyme_map.loc[:,["Enzymes","Type"]].drop_duplicates().set_index("Enzymes").squeeze().to_dict()

    for enzyme in SC_wide.index:
        dicti["UniProtID"].append(enzyme)
        dicti["Name"].append(name_lookup[enzyme])
        if enzyme in cluster_lookup.keys():
            cluster = cluster_lookup[enzyme]
            dicti["Cluster"].append(cluster)
            dicti["Cluster_num"].append(groups_inv[cluster])
        else:
            dicti["Cluster"].append("Not limiting")
            dicti["Cluster_num"].append("-")
        dicti["Subsystem"].append(subsystem_lookup[enzyme])
        if enzyme in isoenzyme_lookup.keys():
            dicti["Isoenzyme"].append(isoenzyme_lookup[enzyme])
        else:
            dicti["Isoenzyme"].append("No complex")

    pd.DataFrame(dicti).to_csv(clust_out_path,index=False)
    print(f"Clustering results were saved at:")
    print({clust_enrichment_out})
    print()

    ## Isoenzyme counts
    print()
    print("== ISOENZYME PROPORTION ==")
    isoenzyme_map = isoenzyme_map.loc[:,["Enzymes","Type"]].drop_duplicates()
    print("Proportion of isoenzymes in each cluster")
    for i in np.arange(4):
        print(f"Cluster {i+1}:")
        cluster_enzymes = full_df[group_assignment==i].index.tolist()
        ## Isoenzymes
        isIso = pd.Series(cluster_enzymes).isin(isoenzyme_map["Enzymes"])
        n = len(cluster_enzymes)
        n_Iso = sum(isIso)
        n_nonIso = n- n_Iso
        print(f"Non-isoenzymes: {n_nonIso} - {n_nonIso / n * 100}")
        print(f"Isoenzymes: {n_Iso} - {n_Iso / n * 100}")
        print(isoenzyme_map.loc[isoenzyme_map["Enzymes"].isin(cluster_enzymes),"Type"].value_counts())
        print()

    print()
    print("== SUBSYSTEM ENRICHMENT ==")
    # Subsystem enrichment
    dicti = {"Cluster":[],"Subsystem":[],"Statistic":[],"P-value":[]}
    removed_subsystems = []
    minimum_enzymes_in_subsystem = 3
    for i in np.arange(4):
        print(f"Cluster {i+1}:")
        cluster_enzymes = full_df[group_assignment==i].index.tolist()
        
        print(subsystem_map[subsystem_map["Enzyme"].isin(cluster_enzymes)].Subsystem.value_counts().head())
        print()
        for sub in subsystem_map["Subsystem"].unique():
                    enz_in_sub = subsystem_map.loc[subsystem_map["Subsystem"]==sub,"Enzyme"].unique()
                    N = 671
                    K = len(enz_in_sub)
                    if(K<minimum_enzymes_in_subsystem):
                            removed_subsystems.append(sub)
                            continue
                    n = len(cluster_enzymes)
                    k = len(np.intersect1d(enz_in_sub,cluster_enzymes))
                    ctab = [[k , n-k],
                            [K-k, N-K-n+k]]
                    res = sp.fisher_exact(ctab, alternative="greater")
                    dicti["Cluster"].append(i+1)
                    dicti["Subsystem"].append(sub)
                    dicti["Statistic"].append(res.statistic)
                    dicti["P-value"].append(res.pvalue)
        
        
    test_df = pd.DataFrame(dicti)

    test_df["FDR_adj_p"] = multitest.multipletests(test_df["P-value"],alpha=0.05,method="fdr_bh")[1]

    test_df.to_csv(clust_enrichment_out,index=False)
    print(f"Enrichment results for the clusters were saved at:")
    print({clust_enrichment_out})


    print()
    print("== PROPORTION OF SUBSYSTEM IN CLUSTER ==")
    dicti = {"Subsystem":[],
             "n_enzymes":[],
             "n_LE":[],
             "Cluster":[],
             "ProportionOfLE":[]}
    for sub in subsystem_map["Subsystem"].unique():
         enz_in_sub = subsystem_map.loc[subsystem_map["Subsystem"]==sub,"Enzyme"].unique()
         active_in_sub = np.intersect1d(enz_in_sub, full_df.index)
         #print(f"Sub - {sub},\t n_enz - {len(enz_in_sub)},\t n_act - {len(active_in_sub)}")
         for clust in np.arange(4):
              cluster_enzymes = full_df[group_assignment==clust].index.tolist()  
              
              in_clust_and_sub = np.intersect1d(active_in_sub,cluster_enzymes)
              dicti["Subsystem"].append(sub)
              dicti["n_enzymes"].append(len(enz_in_sub))
              dicti["n_LE"].append(len(active_in_sub))
              dicti["Cluster"].append(groups[clust])
              
              dicti["ProportionOfLE"].append(len(in_clust_and_sub) / len(active_in_sub) if  len(active_in_sub)!=0 else 0)

    df = (pd.DataFrame(dicti).
     pivot(index=["Subsystem","n_enzymes","n_LE"],columns = "Cluster",values="ProportionOfLE").
     sort_values([v for k,v in groups.items()],ascending=False).
     reset_index().
     set_index("Subsystem"))
    df.to_csv(clust_subs_prop_out)
    
    print(f"Proportional results for the clusters were saved at:")
    print({clust_subs_prop_out})

    fig, ax = plt.subplots(nrows=7,ncols=3,sharex=True,sharey=True,constrained_layout=True,figsize=(10,7))
    # Find top 9 largest subsystems
    rel_subs = df.sort_values("n_LE",ascending=False).head(21).index
    # For each subsystem 
    df = pd.DataFrame(dicti)
    for i in np.arange(len(rel_subs)):
        axes = ax.flat[i]
        data = df.loc[df["Subsystem"]==rel_subs[i],:]
        #print(f"{rel_subs[i]} : {data.shape}")
        sns.barplot(data,x="ProportionOfLE",
                    y ="Cluster",
                    hue = "Cluster",
                    ax=axes,orient="h",
                    palette={groups[k]:c for k,c in colormap.items()},
                    legend=True
                    )
        if i==0:
            handles, labels = axes.get_legend_handles_labels()
        axes.get_legend().remove()
        axes.set_xlim((0,1))
        axes.set_title(rel_subs[i]+ f" ({data.loc[:,"n_LE"].iloc[0]}/{data.loc[:,"n_enzymes"].iloc[0]})")
        axes.set_ylabel("")
        axes.get_yaxis().set_visible(False)
        
    fig.legend(handles,labels,loc="center left",bbox_to_anchor=(1,0.5))

    fig.savefig(supp_fig_out_path, dpi=300, bbox_inches='tight')

    plt.show(block=SHOW_FIGURES)

   

    
if __name__ == "__main__":
    main()