# -*- coding: utf-8 -*-
"""
2023/2/7

@author: T.Iwasaka
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

from statistics import stdev

warnings.simplefilter('ignore')

def vectorization(df_z):
    timelist = [3, 6, 9, 24, 96, 192, 360, 696]
    alttimelist = ["ALT_"+str(i) for i in timelist]
    asttimelist = ["AST_"+str(i) for i in timelist]
    alttimelist[len(alttimelist):len(alttimelist)] = asttimelist
    mat = np.zeros_like(alttimelist)
    mat[0:16] = np.nan

    complst = [i.split("_")[0]+"_"+ i.split("_")[1] for i in df_z.index.tolist() if i.split("_")[1] == "High"] # High concentration
    complst = sorted(set(complst), key=complst.index)
    df0 = pd.DataFrame(index=['ALT_3','ALT_6','ALT_9','ALT_24','ALT_96','ALT_192','ALT_360','ALT_696','AST_3','AST_6','AST_9','AST_24','AST_96','AST_192','AST_360','AST_696'], columns=complst)

    for comp in complst:
        # 化合物の抽出
        if comp == "1% cholesterol + 0.25% sodium cholate_High":
            df_z_comp = df_z[df_z.index.str.contains("cholate_High")]
        elif comp == "acetamide_High":
            df_z_comp = df_z[df_z.index.str.contains(comp)]
            df_z_comp = df_z_comp[~df_z_comp.index.str.contains("thioacetamide")]
        else:
            df_z_comp = df_z[df_z.index.str.contains(str(comp))]
        # 並び替え
        sorter = [(float(i.split("_")[2]), i) for i in set(df_z_comp.index.tolist())] #setにすることでindexの重複を防ぐ
        sorter = sorted(sorter)
        alt_ast = df_z_comp.loc[[i[1] for i in sorter], :]
        sort_list = sorted(set(alt_ast.index.tolist()), key=alt_ast.index.tolist().index)
        
        # 空の辞書の作成
        vect_dict = dict(zip(alttimelist, mat))
        # 値の追加
        biochem_val = "ALT(IU/L)"
        for i in sort_list:
            for j in timelist:
                if int(i.split("_")[2]) == int(j):
                    vect_dict["ALT_"+str(i.split("_")[2])] = alt_ast.loc[i, biochem_val].mean()
                else:
                    pass

        biochem_val = "AST(IU/L)"
        for i in sort_list:
            for j in timelist:
                if int(i.split("_")[2]) == int(j):
                    vect_dict["AST_"+str(i.split("_")[2])] = alt_ast.loc[i, biochem_val].mean()
                else:
                    pass
        
        # データフレームへ反映
        df0[comp] = vect_dict.values()
        df0.replace('nan', np.nan, inplace=True)

    return df0.T

def annotation_sample(df, df_b, check_code=False):
    """
    Annotating the index (or columns) of the data frame
    
    Parameters
    ----------
    df: dataframe
        a dataframe to be analyzed

    df_b: dataframe
        a dataframe with information to be annotated
    
    check_code: bool, default False
        Put the code at the end of the list
    
    Returns
    ----------
    ind: list, str
    
    """
    temp = df_b.loc[df.columns.tolist()]
    name = temp["COMPOUND_NAME"].tolist()
    dose = temp["DOSE_LEVEL"].tolist()
    time = temp["SACRIFICE_PERIOD"].tolist()
    barcode = temp.index.tolist()
    if check_code:
        ind = [f"{i}_{j}_{k}.{l}" for i, j, k, l in zip(name, dose, time, barcode)]
    else:
        ind = [f"{i}_{j}_{k}" for i, j, k in zip(name, dose, time)]
    return ind

def array_imputer(df,threshold=0.9,strategy="median",trim=1.0,batch=False,lst_batch=[], trim_red=True):
    """
    imputing nan and trim the values less than 1
    
    Parameters
    ----------
    df: dataframe
        a dataframe to be analyzed
    
    threshold: float, default 0.9
        determine whether imupting is done or not dependent on ratio of not nan
        
    strategy: str, default median
        indicates which statistics is used for imputation
        candidates: "median", "most_frequent", "mean"
    
    lst_batch: list, int
        indicates batch like : 0, 0, 1, 1, 1, 2, 2
    
    Returns
    ----------
    df_res: dataframe
    
    """
    if (type(trim)==float) or (type(trim)==int):
        df = df.where(df > trim)
    else:
        pass
    df = df.replace(0,np.nan)
    if batch:
        lst = []
        ap = lst.append
        for b in range(max(lst_batch)+1):
            place = [i for i, x in enumerate(lst_batch) if x == b]
            print("{0} ({1} sample)".format(b,len(place)))
            temp = df.iloc[:,place]
            if temp.shape[1]==1:
                ap(pd.DataFrame(temp))
            else:
                thresh = int(threshold*float(len(list(temp.columns))))
                temp = temp.dropna(thresh=thresh)
                imr = SimpleImputer(strategy=strategy)
                imputed = imr.fit_transform(temp.values.T) # impute in columns
                ap(pd.DataFrame(imputed.T,index=temp.index,columns=temp.columns))
        if trim_red:
            df_res = pd.concat(lst,axis=1)
            df_res = df_res.replace(np.nan,0) + 1
            print("redundancy trimming")
        else:
            df_res = pd.concat(lst,axis=1,join="inner")
    else:            
        thresh = int(threshold*float(len(list(df.columns))))
        df = df.dropna(thresh=thresh)
        imr = SimpleImputer(strategy=strategy)
        imputed = imr.fit_transform(df.values.T) # impute in columns
        df_res = pd.DataFrame(imputed.T,index=df.index,columns=df.columns)
    print("strategy: {}".format(strategy))
    return df_res

def batch_norm(df,lst_batch=[]):
    """
    batch normalization with combat
    
    Parameters
    ----------
    df: a dataframe
        a dataframe to be analyzed
    
    lst_batch : lst, int
        indicates batch like : 0, 0, 1, 1, 1, 2, 2
    
    """
    from combat.pycombat import pycombat

    comb_df = pycombat(df,lst_batch)
    return comb_df

def quantile(df,method="median"):
    """
    quantile normalization of dataframe (feature x sample)
    
    Parameters
    ----------
    df: dataframe
        a dataframe subjected to QN (feature x sample)
    
    method: str, default "median"
        determine median or mean values are employed as the template

    Returns
    ----------
    df2: dataframe

    """
    df_c = df.copy() # deep copy
    idx = list(df_c.index)
    col = list(df_c.columns)
    n_idx = len(idx)

    ### prepare mean/median distribution
    x_sorted = np.sort(df_c.values,axis=0)[::-1]
    if method=="median":
        temp = np.median(x_sorted,axis=1)
    else:
        temp = np.mean(x_sorted,axis=1)
    temp_sorted = np.sort(temp)[::-1]

    ### prepare reference rank list
    x_rank_T = df_c.rank(method="first").T.values.astype(int)

    ### conversion
    rank = sorted([int(v + 1) for v in range(n_idx)],reverse=True)
    converter = dict(list(zip(rank,temp_sorted)))
    converted = []
    converted_ap = converted.append
    for arr in tqdm(x_rank_T):
        tra = [converter[v] for v in arr]
        converted_ap(tra)
    np_data = np.array(converted).T
    df2 = pd.DataFrame(np_data,index=idx,columns=col)
    return df2

def calc_z(res, drugs, control="Control"):
    """
    Assuming the control group (for each compound) is the population, zscore is calculated
    
    Parameters
    ----------
    res: dataframe
        a dataframe to be calculated (sample x feature)
    
    drugs: list, str
        list of compound names

    control: str, default "Control"
        specify a name for the control group

    Returns
    ----------
    df_z: dataframe

    """
    df_z = pd.DataFrame(columns=res.columns.tolist())
    for drug in drugs:
        if drug == "acetamide":
            res_temp = res.loc[res.index.str.contains(drug),:]
            res_temp = res_temp.loc[~res_temp.index.str.contains("thioacetamide"),:]
        else:
            res_temp = res.loc[res.index.str.contains(drug),:]
        res_ctrl = res_temp.loc[res_temp.index.str.contains(control),:]
        res_drug = res_temp.loc[~res_temp.index.str.contains(control),:]
        mean = np.nanmean(res_ctrl.values,axis=0)
        std = np.nanstd(res_ctrl.values,axis=0)
        res_z = pd.DataFrame((res_drug.values - mean)/std)
        res_z.index = res_drug.index
        res_z.columns = res_drug.columns
        res_z = res_z.replace(np.inf,0)
        res_z = res_z.replace(-np.inf,0)
        res_z = res_z.fillna(0)
        if len(res.index)!=0:
            df_z = pd.concat([df_z, res_z],axis=0)
    return df_z

def calc_z_all(res, drugs, control="Control"):
    """
    Assuming the control group (for all compounds) is the population, zscore is calculated
    
    Parameters
    ----------
    res: dataframe
        a dataframe to be calculated (sample x feature)
    
    drugs: list, str
        list of compound names

    control: str, default "Control"
        specify a name for the control group

    Returns
    ----------
    df_z: dataframe

    """
    df_z = pd.DataFrame(columns=res.columns.tolist())
    res_ctrl = res.loc[res.index.str.contains(control),:]
    mean = np.nanmean(res_ctrl.values,axis=0)
    std = np.nanstd(res_ctrl.values,axis=0)

    for drug in drugs:
        if drug == "acetamide":
            res_temp = res.loc[res.index.str.contains(drug),:]
            res_temp = res_temp.loc[~res_temp.index.str.contains("thioacetamide"),:]
        else:
            res_temp = res.loc[res.index.str.contains(drug),:]
        res_drug = res_temp.loc[~res_temp.index.str.contains(control),:]
        res_z = pd.DataFrame((res_drug.values - mean)/std)
        res_z.index = res_drug.index
        res_z.columns = res_drug.columns
        res_z = res_z.replace(np.inf,0)
        res_z = res_z.replace(-np.inf,0)
        res_z = res_z.fillna(0)
        if len(res.index)!=0:
            df_z = pd.concat([df_z, res_z],axis=0)
    return df_z

def consensus_sig(data, sep="_", position=1):
    """
    to generate consensus signature
    by linear combination with weightning Spearman correlation
    
    Parameters
    ----------
    data: a dataframe
        a dataframe to be analyzed
    
    sep: str, default "_"
        separator for sample name
        
    position: int, default 1
        indicates position of sample name such as drug    
    
    """
    col = list(data.columns)
    ind = list(data.index)
    rep = [v.split(sep)[position] for v in col]
    rep_set = list(set(rep))
    rank = data.rank()
    new_col = []
    res = []
    ap = res.append
    for r in rep_set:
        mask = [i for i,v in enumerate(rep) if r==v]
        new_col += [col[i] for i,v in enumerate(rep) if r==v]
        temp = data.iloc[:,mask].values.T # check, to_numpy()
        if len(mask) > 2:
            temp_rank = rank.iloc[:,mask].values.T # check, to_numpy()
            corr = np.corrcoef(temp_rank)
            corr_sum = np.sum(corr,axis=1) - 1
            corr = corr/np.c_[corr_sum]
            lst = []
            for j in range(corr.shape[0]):
                temp2 = np.delete(temp,j,axis=0)
                corr2 = np.delete(corr[j],j)
                lst.append(np.dot(corr2,temp2))
            ap(np.array(lst))
        else:
            
            ap(temp)
    res = np.concatenate(res,axis=0).T
    df = pd.DataFrame(res,index=ind,columns=new_col)
    df = df.loc[:,col]
    return df

def plot(means, bar, summary, ax, r, color_a="firebrick", color_b="black", l="liver"):
    # barplot
    error_bar_set = dict(lw=1, capthick=2, capsize=12)
    ax.bar(r, means, yerr=bar, width=0.4, color="w", edgecolor="w", error_kw=error_bar_set, alpha=0.8)
    # line plot
    ax.plot(r, means, color=color_a, linewidth=2, label=l, alpha=0.8)
    # dot plot
    for (lis, numb) in zip(summary, r):
        if type(lis) == np.array([], dtype=np.float64).dtype:
            x = [numb]*1
            ax.plot(x, [float(lis)], marker="o", color=color_b, markersize=6, linestyle="None", alpha=0.8)
        else:
            x = [numb]*len(lis)
            ax.plot(x, [float(i) for i in lis], marker="o", color=color_b, markersize=6, linestyle="None", alpha=0.8)

def gene_plot(df_z, genelst, loc="lower left", baseline=0):
    """
    Plot changes in gene expression over time
    
    Parameters
    ----------
    df_z: a dataframe
        a dataframe to be plotted

    genelst: list of two-dimensional arrays (e.g. [["Aldh4a1", "orangered"], ["Maob", "gold"]]), str
        specify gene name and plot color
    
    loc: str, default "lower left"
        the location of the legend
        
    baseline: float, default 0
        indicates position of baseline 
    
    """
    target_compounds = list({i.split("_")[0] for i in df_z.index.tolist()})
    fig = plt.figure(figsize=(10, 5*(len(target_compounds)//2+1)))
    for i, comp in enumerate(target_compounds):
        df_comp = df_z[df_z.index.str.contains(comp)]
        sorter = [(float(i.split("_")[2].split(" h")[0]), i) for i in set(df_comp.index.tolist())] #setにすることでindexの重複を防ぐ
        sorter = sorted(sorter)
        df_sort = df_comp.loc[[i[1] for i in sorter], :]
        df_sort.index = [i.split(".")[0] for i in df_sort.index.tolist()]
        df_sort.head()

        ax = fig.add_subplot(len(target_compounds)//2+1, 2, i+1)

        for gene, color in genelst:
            drugs = df_sort.index.unique()
            summary = [df_sort.loc[drug, gene] for drug in drugs]
            means = [df_sort.loc[drug, gene].mean() for drug in drugs]
            bar = []
            ap = bar.append
            for drug in drugs:
                if type(df_sort.loc[drug, gene]) == np.array([], dtype=np.float64).dtype:
                    ap(0)
                else:
                    ap(df_sort.loc[drug, gene].sem())

            r = range(len(drugs))
            plot(means, bar, summary, ax, r, color_a=color, color_b=color, l=gene)
        ax.hlines(y=baseline, xmin=-1, xmax=len(drugs), color="black", linestyles='dotted', alpha=.5) # threshold line

        ax.set_xticks(range(len(drugs)))
        ax.set_xticklabels([i.split("_")[-1] for i in drugs], rotation=90, fontsize=14)
        ax.set_xlim(-1,len(drugs))
        ax.set_ylabel("Z score", fontsize=16)
        ax.set_xlabel("Time", rotation=0, fontsize=16)
        ax.set_title(comp, fontsize=18)
        ax.legend(loc=loc)
        plt.yticks(fontsize=14)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')

        plt.grid(False)

    plt.tight_layout()
    plt.show()

def total_plot(means, bar, summary, ax, r, color_a="firebrick", color_b="black", l="liver"):
    # barplot
    error_bar_set = dict(lw=1, capthick=2, capsize=12)
    ax.bar(r, means, yerr=bar, width=0.4, color="w", edgecolor="w", error_kw=error_bar_set, alpha=0.8)
    # line plot
    led = ax.plot(r, means, color=color_a, linewidth=2, label=l, alpha=0.8)
    ax.plot(r, means, color=color_a, linewidth=2, label=l, alpha=0.8)
    # dot plot
    for (lis, numb) in zip(summary, r):
        if type(lis) == np.array([], dtype=np.float64).dtype:
            x = [numb]*1
            ax.plot(x, [float(lis)], marker="o", color=color_b, markersize=6, linestyle="None", alpha=0.8)
        else:
            x = [numb]*len(lis)
            ax.plot(x, [float(i) for i in lis], marker="o", color=color_b, markersize=6, linestyle="None", alpha=0.8)
    return led

def gene_alt_plot(df_z, df_altast, genelst, plus=False):
    """
    Plot changes in gene expression & ALT over time
    
    Parameters
    ----------
    df_z: a dataframe
        a dataframe to be plotted (gene expression)

    df_z: a dataframe
        a dataframe to be plotted (ALT value)

    genelst: list of two-dimensional arrays (e.g. [["Aldh4a1", "orangered"], ["Maob", "gold"]]), str
        specify gene name and plot color
        
    plus: bool, default False
        True if gene expression is elevated
    
    """
    target_compounds = list({i.split("_")[0] for i in df_z.index.tolist()})
    fig = plt.figure(figsize=(10, 5*(len(target_compounds)//2+1)))

    for i, comp in enumerate(target_compounds):
        ax = fig.add_subplot(len(target_compounds)//2+1, 2, i+1)
        ax2 = ax.twinx()

        # GENE PLOT

        # sort
        df_comp = df_z[df_z.index.str.contains(comp)]
        sorter = [(float(i.split("_")[2].split(" h")[0]), i) for i in set(df_comp.index.tolist())] #setにすることでindexの重複を防ぐ
        sorter = sorted(sorter)
        df_sort = df_comp.loc[[i[1] for i in sorter], :]
        df_sort.index = [i.split(".")[0] for i in df_sort.index.tolist()]
        df_sort.head()

        ledlst = []
        apled = ledlst.append
        min1 = 0
        max1 = 0
        for gene, color in genelst:
            drugs = df_sort.index.unique()
            summary = [df_sort.loc[drug, gene] for drug in drugs]
            means = [df_sort.loc[drug, gene].mean() for drug in drugs]
            if min(means) < min1:
                min1 = min(means)
            if max(means) > max1:
                max1 = max(means)
            bar = []
            ap = bar.append
            for drug in drugs:
                if type(df_sort.loc[drug, gene]) == np.array([], dtype=np.float64).dtype:
                    ap(0)
                else:
                    ap(df_sort.loc[drug, gene].sem())


            r = range(len(drugs))
            led = total_plot(means, bar, summary, ax, r, color_a=color, color_b=color, l=gene)
            apled(led[0])

        # ALT PLOT
        biochem_val = "ALT(IU/L)"
        color = "gray"
        conc = "High"

        df_comp = df_altast[df_altast.index.str.contains(comp)]
        df_comp = df_comp[df_comp.index.str.contains(conc)]
        df_comp.index = [i.split("_")[0]+"_"+df_comp.loc[:, "time"][i] for i in df_comp.index.tolist()]

        sorter = [(float(i.split("_")[1].split(" h")[0]), i) for i in set(df_comp.index.tolist())] #setにすることでindexの重複を防ぐ
        sorter = sorted(sorter)
        df_sort = df_comp.loc[[i[1] for i in sorter], :]
        df_sort.index = [i.split(".")[0] for i in df_sort.index.tolist()]
        df_sort.head()

        min2 = 0
        max2 = 0
        drugs = df_sort.index.unique()
        summary = [df_sort.loc[drug, biochem_val] for drug in drugs]
        means = [df_sort.loc[drug, biochem_val].mean() for drug in drugs]
        if min(means) < min2:
                min2 = min(means)
        if max(means) > max2:
            max2 = max(means)
        bar = []
        ap = bar.append
        for drug in drugs:
            if type(df_sort.loc[drug, biochem_val]) == np.array([], dtype=np.float64).dtype:
                ap(0)
            else:
                ap(df_sort.loc[drug, biochem_val].sem())

        r = range(len(drugs))
        led2 = total_plot(means, bar, summary, ax2, r, color_a=color, color_b=color, l=biochem_val.split("(")[0])
        apled(led2[0])

        ax.hlines(y=0, xmin=-1, xmax=len(drugs), color="black", linestyles='dotted', alpha=.8) # threshold line
        ax2.hlines(y=0, xmin=-1, xmax=len(drugs), color="black", linestyles='dotted', alpha=.8) # threshold line

        if abs(min1) > abs(max1):
            lim = abs(min1)
        else:
            lim =  abs(max1)
        if abs(min2) > abs(max2):
            lim2 = abs(min2)
        else:
            lim2 =  abs(max2)

        if plus:
            lim = (lim//2)*2+3
            lim2 = ((lim2//10)*10+50)*2
            ax.set_ylim(-lim,lim)
            ax2.set_ylim(-5,lim2)
            
        else:
            lim = (lim//2)*2+3
            lim2 = (lim2//10)*10+50
            ax.set_ylim(-lim,lim)
            ax2.set_ylim(-lim2,lim2)

        ax.set_xticks(range(len(drugs)))
        ax.set_xticklabels([i.split("_")[-1] for i in drugs], rotation=90, fontsize=14)
        ax.set_xlim(-1,len(drugs))
        ax.set_ylabel("gene Z score (/)", fontsize=14)
        ax2.set_ylabel('ALT (U/L)', rotation=270, fontsize=14)
        ax.set_xlabel("Time", rotation=0, fontsize=16)
        ax.set_title(comp, fontsize=18)

        ax.legend(handles=ledlst, loc="upper left")

        plt.grid(False)
        plt.tight_layout()

    plt.tight_layout()
    plt.show()

def degcheck(df, complst, n=5, xlim=15, ylim=7, method="", threshold=0.05, alternative="less", gn="", gfn="", deg_all=False):
    genename = df.index.tolist()
    print("Number of genes :", len(genename), "\n")

    if method == "bonferroni":
        print("method : Bonferroni")
        print("p value =", threshold, "\n")
        
    elif method == "BH":
        print("method : Benjamini & Hochberg")
        print("q value =", threshold, "\n")

    else:
        print("without multiple testing correction")
        print("p value =", threshold, "\n")
        pass

    fig = plt.figure(figsize=(14, 6.5*(len(complst)//2 + len(complst)%2)))

    degup = []
    addegup = degup.append
    degdown = []
    addegdown = degdown.append

    print(complst[0][1]+" -> "+complst[0][2], "\n")

    if gn == "":
        pass
    elif gn not in genename:
        print("ERROR :", gn, "does not exist.")
        gn = ""
    else:
        pass

    for l, drug in enumerate(complst):
        if deg_all:
            df_24 = df.T.loc[df.T.index.str.contains(str(complst[0][1])),:]
            df_192 = df.T.loc[df.T.index.str.contains(str(complst[0][2])),:]
            df_a_z = df_24
            df_b_z = df_192
        else:
            df_temp9 = df.T.loc[df.T.index.str.contains(str(drug[1])),:]
            df_temp24 = df.T.loc[df.T.index.str.contains(str(drug[2])),:]
            df_a_z = df_temp9.loc[df_temp9.index.str.contains(drug[0]),:]
            df_b_z = df_temp24.loc[df_temp24.index.str.contains(drug[0]),:]

        if method == "bonferroni":
            plim = threshold/len(genename)

        elif method == "BH":
            # Welch's T-Test
            t_all, p_all =stats.ttest_ind(df_a_z, df_b_z, alternative=alternative, equal_var=False)

            m = len(genename)
            p_sort = np.sort(p_all) # sort
            i = m

            # BH method
            while m*p_sort[i-1]/i > threshold:
                if i > 1:
                    i = i-1
                else:
                    break
                
            if (i == 1) & (m*p_sort[i-1]/i > threshold) :
                plim = 0
            else:
                plim = p_sort[i-1]

        else:
            plim = threshold
            pass

        fclst = []
        plst = []
        genlst1 = []
        genlst2 = []
        apfc = fclst.append
        app = plst.append
        apg1 = genlst1.append
        apg2 = genlst2.append

        fclst2 = []
        plst2 = []
        apfc2 = fclst2.append
        app2 = plst2.append

        for j in genename:
            if j == gn:
                fc1 = (df_b_z[j].mean())-(df_a_z[j].mean())
                apfc(fc1)
                t1, p1 =stats.ttest_ind(df_a_z[j], df_b_z[j], alternative='two-sided', equal_var=False)
                app(-np.log10(p1))

                if (fc1 > n) & (p1 < plim):
                    apg1(j)
                elif (fc1 < -n) & (p1 < plim):
                    apg2(j)

            elif gfn in j:
                fc2 = (df_b_z[j].mean())-(df_a_z[j].mean())
                apfc(fc2)
                apfc2(fc2)
                t2, p2 =stats.ttest_ind(df_a_z[j], df_b_z[j], alternative='two-sided', equal_var=False)
                app(-np.log10(p2))
                app2(-np.log10(p2))

                if (fc2 > n) & (p2 < plim):
                    apg1(j)
                elif (fc2 < -n) & (p2 < plim):
                    apg2(j)

            else:
                fc = (df_b_z[j].mean())-(df_a_z[j].mean())
                apfc(fc)
                t, p =stats.ttest_ind(df_a_z[j], df_b_z[j], alternative='two-sided', equal_var=False)
                app(-np.log10(p))

                if (fc > n) & (p < plim):
                    apg1(j)
                elif (fc < -n) & (p < plim):
                    apg2(j)


        addegup(genlst1)
        addegdown(genlst2)        
        ax = fig.add_subplot((len(complst)//2 + len(complst)%2), 2, l+1)
        ax.set_xlim([-xlim,xlim])
        ax.set_ylim([-0.3,ylim])
        ax.axvspan(n, xlim, color = "coral", alpha=0.5)
        ax.axvspan(-xlim, -n, color = "coral", alpha=0.5)
        ax.axhspan(-np.log10(plim), ylim, color = "coral", alpha=0.5)
        ax.scatter(fclst, plst)
        if gfn != "":
            ax.scatter(fclst2, plst2, color="gold")
        if gn != "":
            ax.scatter(fc1, -np.log10(p1), color = "red")
        ax.axvline(n, ls = "-.", color = "gray")
        ax.axvline(-n, ls = "-.", color = "gray")
        ax.axhline(-np.log10(plim), ls = "-.", color = "gray")

        plt.title(drug[0])
        plt.xlabel("FC")
        plt.ylabel("-log(p)")

        if deg_all:
            break

    return degup, degdown

def calc_stat(df, timelst):
    time_point = ["_"+str(i)+" hr" for i in timelst]
    df_res = pd.DataFrame(index=df.columns, columns=time_point)
    for time in time_point:
        for target in df.columns.tolist():
            values = df.loc[df.index.str.contains(time),target].values.flatten()
            v_mean = np.mean(values)
            v_conf = stats.t.interval(alpha=0.95, df=len(values)-1, loc=v_mean, scale=stats.sem(values))
            v_sd = stdev(values)
            df_res.loc[target, time]=[v_mean, v_conf[0], v_conf[1], v_sd]
    return df_res

def average_moving(df_stat, title, genelst, timelst, x_equal_pace=False, type="95%", sep=3, y_lim=0):
    """
    Plot changes in gene expression mean & 95% CI (or sd) over time
    
    Parameters
    ----------
    df_stat: a dataframe
        a dataframe to be plotted (gene expression mean, 95% CI and sd)

    gene_lst: list, str
        specify gene name
        
    timelst: list, int
        specify time
    
    """
    time_point = ["_"+str(i)+" hr" for i in timelst]
    df = df_stat.T
    df_target = pd.DataFrame()
    for time in time_point:
        df_time = df[df.index.str.contains(time)]
        df_target = pd.concat([df_target, df_time])
    df_target = df_target.T
    fig = plt.figure(figsize=(4*sep, 4*(len(genelst)//sep+1)))
    for i, gene in enumerate(genelst):
        ax = fig.add_subplot(len(genelst)//sep+1, sep, i+1)
        v_mean = [i[0] for i in df_target.loc[gene,:].tolist()]
        v_upper = [i[1] for i in df_target.loc[gene,:].tolist()]
        v_lower = [i[2] for i in df_target.loc[gene,:].tolist()]
        v_sd = [i[3] for i in df_target.loc[gene,:].tolist()]
        v_sd_p = [x+y for (x, y) in zip(v_mean, v_sd)]
        v_sd_m = [x-y for (x, y) in zip(v_mean, v_sd)]

        if x_equal_pace:
            if type == "95%":
                ax.fill_between(range(len(time_point)), v_lower, v_upper, color="lightskyblue", alpha=0.4)
            elif type == "sd":
                ax.fill_between(range(len(time_point)), v_sd_m, v_sd_p, color="paleturquoise", alpha=0.4)
            else:
                pass
            ax.plot(range(len(time_point)), v_mean, color="royalblue")
            ax.scatter(range(len(time_point)), v_mean, color="mediumblue")
            ax.set_xticks(range(len(time_point)))

        else:
            if type == "95%":
                ax.fill_between(timelst, v_lower, v_upper, color='salmon', alpha=0.4)
            elif type == "sd":
                ax.fill_between(timelst, v_sd_m, v_sd_p, color='coral', alpha=0.4)
            else:
                pass
            ax.plot(timelst, v_mean, color="firebrick")
            ax.scatter(timelst, v_mean, color="darkred")
            ax.set_xticks(timelst)
            
        ax.set_xticklabels(timelst, rotation=0, fontsize=11)
        ax.set_xlabel("Time (h)", fontsize=13)
        if y_lim == 0:
            pass
        elif y_lim < 0:
            ax.set_ylim(y_lim, 2.5)
        elif y_lim > 0:
            ax.set_ylim(-2.5, y_lim)
        ax.set_title(gene, fontsize=16)
        fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.show()