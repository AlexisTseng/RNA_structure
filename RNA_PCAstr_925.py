#!/usr/bin/env python3
# script: foldSPS.py
# author: Daniel Desiro', Jiayi Zeng
script_usage="""
usage
    RNA_PCAstr.py -fsa <fasta_genome> -pfx <out_prefix> -psp <SPLASH-table> [options]

version
    RNA_PCAstr.py 0.0.1 (alpha)

dependencies
    Python v3.9.7, Scipy v1.11.2, NumPy v1.22.2, viennarna v2.5.1, Matplotlib v3.5.1, pandas v2.1.0, scikit-learn v1.30, umap-learn v0.5.3, seaborn v0.12.2, hdbscan v0.8.33

description
    This script takes in pre-processed SPLASH results as tsv, then conducts bootstrapping, PCA, UMAP, HDBSCAN to extract major RNA conformations represented in SPLASH as clusters in UMAP projection. The structural information is then returned as a dictionary -- key = name of cluster/conformation, value = junction object list.

################################################################

--prefix,-pfx
    output directory and prefix for result files

--genomeFasta,-fsa
    input genome FASTA file

--parseSPLASHintra,-psp
    input intra-molecular interaction table from parseSPLASH

--parseSPLASHinter,-psi
    input inter-molecular interaction table from parseSPLASH (default: )

--sampleSize,-sps
    the total number of structure generated and used for PCA (default: 1000)

--segment,-seg
    the segment 

--samplingAttempt,-sat
    the number of sampling attempts from the junction file used to generate one single structure in PCA (default: 10000)

--matrixVal,-mvl
    Value input into PCA matrix. Can be minimal free energy (0, mfe) or binary (0, 1)

--NumComponents,-ncp
    if integer: Number of PC returned from PCA. If fraction: return PCs required to explain this fraction of total variance (default: 0.9)

--nNeighbors,-nnb
    number of neighbours parameter used for UMAP (default: 15)

--minDistance,-mdi
    min distance parameter used for UMAP, can range between 0.0 and 0.99 (default:0.1)    

--minSamp,-msp
    HDBSCAN parameter, minimal sample size. It provides a measure of how conservative you want you clustering to be. The larger the value of min_samples you provide, the more conservative the clustering – more points will be declared as noise. (default: 15)

--minClusterSize,-mcs
    HDBSCAN parameter. The smallest size grouping that you wish to consider a cluster (default: 15)

--junctionRange,-jcg 
    define the maximum distance of nucleotides for a base pair to be connected
    (default: 5)

--junctionCount,-jcc
    define the mean threshold for combining junction base pairs (default: 0.5)

################################################################################

--reverse,-rev
    creates reverse of each sequence in the genome file (default: False)

--complement,-cmp
    creates complements of each sequence in the genome file (default: False)

################################################################################

--viennaRNAdangles,-vrd
    viennaRNA dangling ends setting for RNA folding (default: 2) (choices: 0,1,2,3)

--viennaRNAtemperature,-vrt
    viennaRNA temperature setting for RNA folding (default: 37.0)

--viennaRNAnoLP,-vrn
    viennaRNA no lonely pair setting for RNA folding (default: True)

################################################################################

--newVariable,-nva
    thins thing does sth (default: 4.6)
    
--newVariable2,-nvb
    thins thing does sth (default: 4.0)

################################################################################

--threads,-thr
    number of threads to use for RNAcofold (default: 1)

--overwrite,-ovr
    overwrite data with folder named in prefix (default: False)

--pickleData,-pcl
    load from pickle file if available for faster setting changes 
    (default: False)

--junctionHeader,-jch
    define the junction output header and table data columns
    (default: aSeq,ai,aj,bSeq,bi,bj,jCount,jPeak,raw,jType,MFE,RNA,pattern)

--segmentName,-sgn
    trimm start of segment name (default: 5)

--colorSettings,-cls
    colors for all gradient plotting functions
    (default: #ffffff,#e60202,#222f5b)

################################################################################

--plotHeats,-plh
    plot major interaction heat maps (default: False)

--plotAllHeats,-pla
    plot a interaction heat map for each iteration (default: False)

--plotBasePairs,-plb
    plot base pairs instead of square (default: False)

--plotProcesses,-plp
    turn on multi processing for matrix plotting (default: False)

--plotMatrixMulti,-pll
    turn on multi processing for matrix generation (default: False)

--plotSize,-pls
    defines the plot size modifier (default: 1.0)

--plotMax,-plm
    set maximum heat map plot energy for candidate plots (default: 0.0)

--plotMin,-pli
    set minimum heat map plot energy for candidate plots (default: 0.0)

--plotStepX,-plx
    step size for x axis ticks (default: 100)

--plotStepY,-ply
    step size for y axis ticks (default: 100)

--plotSvg,-plv
    also plot svg heat maps (default: False)

--plotColor,-plc
    reverse plot color (default: False)

--plotNan,-pln
    set zero values to nan (default: False)

--plotThreshold,-plt
    plot threshold for matrix plots (default: 0.0)

################################################################################

reference
    D. Desirò, A. Borodavka.
    "parseSPLASH: ."
    In Preparation, 2022.
    https://github.com/desiro/
"""


## Import
from importlib.metadata import entry_points
import os
from pydoc import describe
import sys
import re
import pickle
import argparse as ap
import time
import pandas as pd
import numpy as np
from numpy import zeros, nan_to_num, arange, nan, ones, mean, median, std, cov, array, quantile, log10, log2, linspace
from operator import attrgetter, itemgetter

try:
    from RNA import fold_compound, cvar, CONSTRAINT_DB, CONSTRAINT_DB_DEFAULT, bp_distance
except:
    print("Error: The mandatory library ViennaRNA 2.5 is missing, please read the README.md for more information!")
    exit()

from random import choices, seed
from sklearn.decomposition import PCA
from sklearn import preprocessing, metrics
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab

from matplotlib.colors import ListedColormap
import multiprocessing as mp
import umap
import seaborn as sns
import hdbscan
import ast
import math
from itertools import combinations

##################################################################
## Main function
##################################################################

def main(opt):
    ####################################################
    ## Step1: make output directory
    pfx_dir = makeDir(opt)
    PCAplotDir, screenDir, PCA_dir, StrBP_dir, PC_LC_Dir, UMAP_dir, pcl_dir= dirGenerate(pfx_dir, opt)
    ####################################################
    ## Step 2: import genome fasta file
    gen_dict = readGenome(opt)
    #print(gen_dict)
    ####################################################
    ## Step 3: mport SPLASH output tsv table
    print("Step 3: Import SPLASH output tsv table")
    junction_list = list()
    segjc_dict = readSPLASHtable(opt, junction_list, gen_dict)

    #print(junction_list[:5])
    #print(all_s1)

    ####################################################
    ## Step 4: Structure Generations by Random Sampling + Storing in DataFrame

    #print("Step 4: Structure Generations by Random Sampling + Storing in DataFrame") 
    #try:
    #    print("loading pickle file")
    #    df, paired_set, strName, len_jc = loadData(f"{pcl_dir}/sampledStructure{opt.pfx}-{opt.seg}")
    #except:
    #    #print(" generating structures and storing as picklefile")
    #    df, paired_set, strName, len_jc = StrGeneration(pfx_dir,segjc_dict, gen_dict, StrBP_dir, opt) #(for one segment, intra)
    #    saveData((df, paired_set, strName, len_jc), f"{pcl_dir}/sampledStructure{opt.pfx}-{opt.seg}")


    ######################################################
    ## Step 5: PCA, visualisation
    
    #print("Step 5: PCA, visualisation and coordinate export")
    #pca, pca_data, per_var, labels = StrPCA(df, PCAplotDir, screenDir, strName, opt)


    ######################################################
    ## Step 6: PC analysis, extract key determinant interactions 
    #print("Step 6: PC analysis, extract key determinant interactions ")
    #PCAcompAna(pca, paired_set, labels, PC_LC_Dir, gen_dict, opt)


    ######################################################
    ## Step 7: Clustering
    print("Step 7: Clustering")

    embedding, clusLabel, clustered, S_score = umapClusterPostPCA_3D(PCAplotDir, UMAP_dir, opt)

    ######################################################
    ## Step 8: Cluster Matrix Extraction

    print("Step 8: Structure Extraction")
    clusMat_dict, clusbpdict_dict, clusteredStr_dict, n = strExtract(pfx_dir, gen_dict, StrBP_dir, clusLabel, clustered, opt)
    

    ######################################################
    ## Step 9: Evaluate quality of structure extraction, output as tsv
    clusEvaluate(gen_dict, clusMat_dict, pfx_dir, S_score, clusteredStr_dict, n, opt)



    ######################################################
    ## Step 10: Cluster Matrix to Structure

    #clusJc_dict = matrix_to_jc(gen_dict, clusbpdict_dict, opt)
    #saveData(clusJc_dict, f"{pcl_dir}/extractedJunctions-{opt.mvl}-{opt.seg}-numNeighbor{opt.nnb}-minDist{opt.mdi}-minSamp{opt.msp}-minCluSize{opt.mcs}-jcRange{opt.jcg}-jcCount{opt.jcc}")



###################################################################
## Operations in Main Function
###################################################################


#####################################################################

def makeDir(opt):
    ## create directory
    dir_name, dir_base = opt.pfx, opt.pfx
    if not opt.ovr:
        i = 1
        while os.path.isdir(dir_name):
            dir_name = f"{dir_base}_{i}"
            i += 1
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    opt.pfx = dir_name
    return dir_name

def loadData(fname): 
    ## load data with pickle
    with open(f"{fname}.pcl", "r+b") as pcl_in:
        pcl_data = pickle.load(pcl_in)
    return pcl_data


def saveData(pcl_data, fname): ## save data with pickle
    with open(f"{fname}.pcl", "w+b") as pcl_out:
        pickle.dump(pcl_data, pcl_out , protocol=4)


def dirGenerate(dir_name, opt):    
    PCA_dir = os.path.join(dir_name,"PCA")
    if not os.path.isdir(PCA_dir):
        os.makedirs(PCA_dir, 0o777)
    UMAP_dir = os.path.join(dir_name, "UMAP-HDBSCAN", f"{opt.seg}")
    if not os.path.isdir(UMAP_dir):
        os.makedirs(UMAP_dir, 0o777)
    screenDir = os.path.join(PCA_dir,"ScreenPlot", f"{opt.seg}")
    if not os.path.isdir(screenDir):
        os.makedirs(screenDir, 0o777)
    PCAplotDir = os.path.join(PCA_dir,"PCAplot", f"{opt.seg}")
    if not os.path.isdir(PCAplotDir):
        os.makedirs(PCAplotDir, 0o777)
    StrBP_dir = os.path.join(PCA_dir,"PCAplot")
    if not os.path.isdir(StrBP_dir):
        os.makedirs(StrBP_dir, 0o777)
    PC_LC_Dir = os.path.join(PCA_dir,"PC_linearCombination")
    if not os.path.isdir(PC_LC_Dir):
        os.makedirs(PC_LC_Dir, 0o777)
    pcl_dir = os.path.join(dir_name,"pcl")
    if not os.path.isdir(pcl_dir):
        os.makedirs(pcl_dir, 0o777)
    return PCAplotDir, screenDir, PCA_dir, StrBP_dir, PC_LC_Dir, UMAP_dir, pcl_dir

def readGenome(opt):
     # load genome fasta file, and return a dictionary
    gen_dict = dict()
    with open(opt.fsa, "r") as genfasta:
        for line in genfasta:
            line = line.strip()
            if re.match(r">", line):
                RNA, name = "", line[1+opt.sgn:].split()[0]
            else:
                RNA += line
            gen_dict[name] = line
        return gen_dict



def readSPLASHtable(opt, junction_list, gen_dict):
    ## read RNA interaction table (for each experiment)
    with open(opt.psp, "r") as inju:
        header = next(inju)
        header = header.strip().split()
        #print(type(header[0]))
        for line in inju:
            #print(f"Status: Read chimeric {i} ...                   ", end="\r")
            line = line.strip().split()
            if re.search(r"[(]|[)]", line[header.index("pattern")]) == None:
                continue
            else:
                chjc = dict()
                for i, item in enumerate(header): 
                    if item == "ai" or item == "bi": chjc[item] = int(line[i])-1
                    elif item == "aj" or item == "bj": chjc[item] = int(line[i])
                    else: chjc[item] = line[i]
                jc = junction(**chjc)
                junction_list.append(jc)
    junction_list = sorted(junction_list, key=attrgetter("jCount"), reverse=True)
    segjc_dict = dict()
    for genome in gen_dict.keys():
        segjc_dict[genome] = [jc for jc in junction_list if jc.aSeq == genome]
    return segjc_dict


def readSPLASHtable_2(opt, junction_list, gen_dict):
    ## read RNA interaction table (for each experiment)
    with open(opt.psp, "r") as inju:
        header = next(inju)
        for line in inju:
            #print(f"Status: Read chimeric {i} ...                   ", end="\r")
            line = line.strip()
            line = line.split()
            #print(line[1])
            if re.search(r"[(]|[)]", line[12]) == None:
                continue
            else:
                chjc = {"aSeq":line[0], "ai":int(line[1])-1, "aj":int(line[2]), "bSeq":line[3], "bi":int(line[4])-1, "bj":int(line[5]), "jCount":line[6],"jPeak":line[7], "raw":line[8],"jType":line[9], "MFE":line[10], "RNA":line[11], "pattern":line[12]}
                jc = junction(**chjc)
                #print(chjc)
                junction_list.append(jc)
    junction_list = sorted(junction_list, key=attrgetter("jCount"), reverse=True)
    segjc_dict = dict()
    for genome in gen_dict.keys():
        segjc_dict[genome] = [jc for jc in junction_list if jc.aSeq == genome]
    return segjc_dict

class junction(object):
    def __init__(self, **data):
        self.__dict__.update((k,self.transf(v)) for k,v in data.items())
    def transf(self, s):
        if isinstance(s, str):
            try: return int(s)
            except ValueError:
                try: return float(s)
                except ValueError:
                    if s in ["True", "False"]: return s == "True"
                    else: return s
        else: return s
    def plot(self, sep):
        ldat = sep.join([f"{var}" for key,var in vars(self).items()])
        return ldat


#############################
#############################


def StrGeneration(pfx_dir,segjc_dict, gen_dict, StrBP_dir, opt):
    ## satRec, satNum = [], [] 
    paired_set, paired, mfeList = set(), [], [] # paired is a list of bplist from all structures
    all_jc, len_jc, segLen = segjc_dict[opt.seg], len(segjc_dict[opt.seg]), len(gen_dict[opt.seg])
    sat = int(0.0003*segLen**2 - 0.485*segLen + 329.78) ## quadratic fitting from mean-sdt over length (sps = 1000, sat = 10k), aim to undersample from SPLASH
    print(f"sat = {sat}")
    for i in range(opt.sps): # generation of one structure
        jcSet = StrSampling_3(all_jc, sat, opt)
        #print(jcSet)
        constr, bplog, jcRecord, bplist = StrCompile(gen_dict, jcSet, opt)
        ## satRec.append(max(jcRecord))
        ## satNum.append(len(jcRecord))
        #print(f"Final constraint String: {list_to_str(constr)} \njcRecord: {jcRecord}  Total number of jc used: {len(jcRecord)}")
        paired_set, paired, mfeList = bpSetGen(paired_set, paired, mfeList, gen_dict, constr, opt) #update paired_set and paired in each run
    #satRec = pd.Series(satRec)
    #print(f"For {opt.seg}: \n{satRec.describe()}")
    paired_set = sorted(list(paired_set))
    print(f"{opt.pfx} {opt.seg}: number of junctions = {len_jc}, len of paired_set: {len(paired_set)}")
    #print(f"paired_set size: {len(paired_set)} \n{paired_set}")
    if opt.mvl == "binary":
        df, paired_set, strName = dfGenerate_2(paired, paired_set, mfeList, StrBP_dir, opt)
    elif opt.mvl == "mfe":
        df, paired_set, strName = dfGenerate(paired, paired_set, mfeList, StrBP_dir, opt)
    else: print("matrixVal input ERROR")
    return df, paired_set, strName, len_jc

#####################
## Sub-function in StrGeneration #Start
def StrSampling(gen_dict, segjc_dict, opt):
    sat = opt.sat
    all_jc = segjc_dict[opt.seg]
    population, weights = [jc for jc in all_jc], [jc.jCount for jc in all_jc]
    jcSet = choices(population, weights, k=sat) # k= size of output list, or number of draws
    return jcSet, sat

def StrSampling_2(gen_dict, segjc_dict, opt):
    segLen = len(gen_dict[opt.seg])
    sat = int(0.0003*segLen**2 - 0.485*segLen + 329.78) ## quadratic fitting from mean-sdt over length (sps = 1000, sat = 10k), aim to undersample from SPLASH
    all_jc = segjc_dict[opt.seg]
    len_jc = len(all_jc)
    population, weights = [jc for jc in all_jc], [jc.jCount for jc in all_jc]
    jcSet = choices(population, weights, k=sat) # k= size of output list, or number of draws
    return jcSet, len_jc

def StrSampling_3(all_jc, sat, opt):
    population, weights = [jc for jc in all_jc], [jc.jCount for jc in all_jc]
    jcSet = choices(population, weights, k=sat) # k= size of output list, or number of draws
    return jcSet

def StrCompile(gen_dict, jcSet, opt):
    jcRecord = []
    constr, bplog = initConsBplog_4(opt,gen_dict) #value = list of list
    for i,jc in enumerate(jcSet):
        bplist = sorted(getBasePairs(jc.ai, jc.bi, jc.pattern))
        test_str, allp = getTestStr(bplist, constr)
        #except:
        #    out_pos = getTestStr_check(bplist, constr)
        #    print(f"out_pos: {out_pos}")
        if isOverlap(test_str) or isPknot_2(bplist, constr, bplog):
            continue
        else:
            constr, bplog = updateStrBp(bplist, constr, bplog)
            #print(f"The {i}th input: \n ai:{jc.ai} \n pattern: {jc.pattern} \n resultant string: {list_to_str(constr)} \n Resultant Bplog: {bplog}")
            jcRecord.append(i)
    return constr, bplog, jcRecord, bplist


def bpSetGen(paired_set, paired, mfeList, gen_dict, constr, opt):
    mfe, pattern = doCofold(gen_dict[opt.seg], list_to_str(constr), opt)
    #print(list_to_str(constr))
    strBP = sorted(getBasePairs(0, -1, pattern))
    #print(f"\n \nconstraint: {list_to_str(constr)} \nPattern   : {pattern} \nStrBP_dir: {strBP}")
    paired_set.update(strBP)
    paired.append(strBP) # paired is a list (structures) of list (bplist)
    mfeList.append(mfe)
    return paired_set, paired, mfeList


def dfGenerate(paired, paired_set, mfeList, StrBP_dir, opt): #mfe
    null, StrListlist, strName  = 0, [], []
    for j, (str, mfe) in enumerate(zip(paired,mfeList)):
        #print(f"\nStructure {j}:")
        dfCol = [null]*len(paired_set)
        for i, bp in enumerate(paired_set):
            if bp in str:
                #print(f" present: bp = {bp} \n Str:{str}")
                dfCol[i] = mfe
            else: 
                #print(f" absent: bp ={bp}")
                continue
        StrListlist.append(dfCol)
        strName.append(f"Str{j}")
        #print(f"{j}: {dfCol}")
    df = pd.DataFrame(StrListlist, columns= paired_set, index= strName)
    df.to_csv(f"{StrBP_dir}/{opt.mvl}-{opt.pfx}-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}.csv")
    #print(df)
    return df, paired_set, strName

def dfGenerate_2(paired, paired_set, mfeList, StrBP_dir, opt):#binary
    null, val = 0, 1
    StrListlist, strName = [], []
    for j, str in enumerate(paired):
        #print(f"\nStructure {j}:")
        dfCol = [null]*len(paired_set)
        for i, bp in enumerate(paired_set):
            if bp in str:
                #print(f" present: bp = {bp} \n Str:{str}")
                dfCol[i] = val
            else: 
                #print(f" absent: bp ={bp}")
                continue
        StrListlist.append(dfCol)
        strName.append(f"Str{j}")
        #print(f"{j}: {dfCol}")
    df = pd.DataFrame(StrListlist, columns= paired_set, index= strName)
    df.to_csv(f"{StrBP_dir}/{opt.mvl}-{opt.pfx}-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}.csv")
    #print(df)
    return df, paired_set, strName

### Sub-functions in StrGenerate, end

################
## Sub-sub functions in StrGenerate, start
def initConsBplog_4(opt,gen_dict):
    ### generate empty constraint string
    const, init = ".", -1
    ### generate empty base paring log dict
    ### Initialise constraints and bplog
    v = gen_dict[opt.seg]
    constr_list = [const]*len(v) # operate as lists, then report as strings
    bplog = [init]*len(v)
    return constr_list, bplog


def initConsBplog_3(opt,gen_dict):
    ### generate empty constraint string
    constr_list, const = [],"."
    ### generate empty base paring log dict
    bplog, init = [], -1
    ### Initialise constraints and bplog
    v = gen_dict[opt.seg]
    constr_list.append([const]*len(v)) # operate as lists, then report as strings
    bplog.append([init]*len(v))
    return constr_list, bplog


def getBasePairs(ai, bi, pattern):
    ## get tuples of base pairs       
    s, stack, pairings = len(pattern.split("&")[0]), list(), list()
    if bi == -1: bi = ai+s+1
    for j,p in enumerate(pattern):
        if p == "(": stack.append(j)
        if p == ")": pairings.append((ai+stack.pop(),bi+j-s-1))
    return set(pairings)

def isOverlap(test_str):
    if re.search(r"[(]|[)]",test_str) == None:
        #print("no overlap")
        ss = 0
    else:
        #print(f"Overlap")
        ss = 1
    return bool(ss)

def getTestStr_check(bplist,constr_list):
    #print(f"constraint string: {list_to_str(constr_list)} \n length:{len(constr_list)}")
    setp1, setp2 = zip(*bplist)
    allp = setp1 + setp2 
    #print(f"setp1: {setp1} \n setp1:{setp2} \n allp:{allp}")
    out_pos = [pos for pos in allp if pos >= len(constr_list) or pos <0]
    return out_pos


def getTestStr(bplist,constr_list):
    #print(f"constraint string: {list_to_str(constr_list)} \n length:{len(constr_list)}")
    setp1, setp2 = zip(*bplist)
    allp = setp1 + setp2 
    #print(f"setp1: {setp1} \n setp1:{setp2} \n allp:{allp}")
    test = [constr_list[pos] for pos in allp]
    test_str = "".join(f"{t}" for t in test)
    return test_str, allp

def updateStrBp(bplist, constr_list, bplog):
    for (p1,p2) in bplist:
        constr_list[p1] = "("
        constr_list[p2] = ")"
        bplog[p1] = p2
        bplog[p2] = p1
    return constr_list, bplog

def list_to_str(list):
    str = "".join(f"{t}" for t in list)
    return str

def doCofold(RNA, constraint, opt):
    ## do Cofold
    cvar.dangles = opt.vrd
    cvar.noLP = int(opt.vrn)
    cvar.temperature = opt.vrt
    fc = fold_compound(RNA)
    fc.constraints_add(constraint, CONSTRAINT_DB | CONSTRAINT_DB_DEFAULT)
    pattern, mfe = fc.mfe()
    return mfe, pattern

def doCofold_2(RNA, opt):
    ## do Cofold
    cvar.dangles = opt.vrd
    cvar.noLP = int(opt.vrn)
    cvar.temperature = opt.vrt
    fc = fold_compound(RNA)
    pattern, mfe = fc.mfe()
    return mfe, pattern


def isPknot_2(bplist,constr_list,bplog):
    e1,s2 = bplist[-1][0], bplist[-1][-1]
    ss = 0
    if [i for i,j in enumerate(bplog[e1+1:s2]) if j != -1]: #bp present
        if e1 < min([j for j in bplog[e1+1:s2] if j != -1]) and s2 > max([j for j in bplog[e1+1:s2] if j != -1]):
            #print(f"Not pseudo knot: {list_to_str(constr_list[e1:s2+1])}")
            #print(f"Not pseudo knot")
            pass
        else:
            #print(f"Is pseudo knot: {list_to_str(constr_list[e1:s2+1])}")
            #print(f"is pseudo knot")
            ss += 1
    else: pass # bp not present
        #print(f"Not pseudo knot: {list_to_str(constr_list[e1:s2+1])}")
        #print(f"Not pseudo knot")
    return bool(ss)

## Sub-function in StrGeneration #End
#########################################



##############################################
##############################################

def StrPCA(df, PCAplotDir, screenDir, strName, opt):
    scaled_df = preprocessing.scale(df)
    pca = PCA(n_components=float(opt.ncp))
    pca.fit(scaled_df) # calculation step
    pca_data = pca.transform(scaled_df) # generate corrdinates for pca graph
    per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
    labels = ["PC" + str(x) for x in range(1, len(per_var)+1)]
    pca_df = pd.DataFrame(pca_data, index= strName, columns=labels)
    print(pca_df)
    pca_df.to_csv(f"{PCAplotDir}/{opt.mvl}PcaCoord-{opt.pfx}-{opt.seg}-sps{opt.sps}.csv")
    ##### screen plot generation
    screenPlot(per_var, labels, screenDir,opt)

    ##### Draw PCA graph and save coordinates
    #PCAplot_3D_mlab(pca_df, per_var, strName, opt)
    #PCAplot_3D(pca_df, per_var, strName, PCAplotDir, opt)
    #PCAplot(pca_df, per_var, strName, PCAplotDir, opt)
    return pca, pca_data, per_var, labels

## Sub-functions in StrPCA, start
def screenPlot(per_var, labels, screenDir,opt):
    scrPlt = plt.figure()
    plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
    plt.xticks(rotation=90, fontsize=4)
    plt.ylabel("Percentage of Explained Variance")
    plt.xlabel("Principal Components")
    plt.title("Screen Plot")
    if opt.mvl == "binary":
        scrPlt.savefig(f"{screenDir}/BinaryScreenPlot-{opt.pfx}-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}.pdf")
    elif opt.mvl == "mfe":
        scrPlt.savefig(f"{screenDir}/MfeScreenPlot-{opt.pfx}-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}.pdf")


def PCAplot_3D_mlab(pca_df, per_var, strName, opt):
    mlab.figure()
    mlab.points3d(pca_df.PC1, pca_df.PC2, pca_df.PC3, scale_factor=0.4)
    mlab.xlabel("PC1 - {0}%".format(per_var[0]))
    mlab.ylabel("PC2 - {0}%".format(per_var[1]))
    mlab.zlabel("PC3 - {0}%".format(per_var[2]))
    if opt.mvl == "binary":
        mlab.title(f"Binary3DPCA-sps{len(strName)}-ncp{opt.ncp}")
        mlab.show()
        #mlab.savefig(f"{PCAplotDir}/Binary3DPcaPlot-{opt.pfx}-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}.pdf")
    elif opt.mvl == "mfe":
        mlab.title(f"Mfe3DPCA-sps{len(strName)}-ncp{opt.ncp}")
        mlab.show()
        #mlab.savefig(f"{PCAplotDir}/Mfe3DPcaPlot-{opt.pfx}-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}.pdf")


def PCAplot_3D(pca_df, per_var, strName, PCAplotDir, opt):
    pcaPlot = plt.figure()
    ax = pcaPlot.add_subplot(111, projection="3d")
    ax.scatter(pca_df.PC1, pca_df.PC2, pca_df.PC3, s=5)
    ax.set_xlabel("PC1 - {0}%".format(per_var[0]))
    ax.set_ylabel("PC2 - {0}%".format(per_var[1]))
    ax.set_zlabel("PC3 - {0}%".format(per_var[2]))
    if opt.mvl == "binary":
        ax.set_title(f"Binary3DPCA-sps{len(strName)}")
        pcaPlot.savefig(f"{PCAplotDir}/Binary3DPcaPlot-{opt.pfx}-{opt.seg}-sps{opt.seg}.pdf")
        pca_df.to_csv(f"{PCAplotDir}/BinaryPcaCoord-{opt.pfx}-{opt.seg}-sps{opt.sps}.csv")
    elif opt.mvl == "mfe":
        ax.set_title(f"Mfe3DPCA-sps{len(strName)}")
        pcaPlot.savefig(f"{PCAplotDir}/Mfe3DPcaPlot-{opt.pfx}-{opt.seg}-sps{opt.seg}-ncp{opt.ncp}.pdf")
        pca_df.to_csv(f"{PCAplotDir}/MfePcaCoord-{opt.pfx}-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}.csv")

def PCAplot(pca_df, per_var, strName, PCAplotDir, opt):
    ## PC1-PC2
    pcaPlot = plt.figure()
    plt.scatter(pca_df.PC1, pca_df.PC2,s=5)
    plt.xlabel("PC1 - {0}%".format(per_var[0]))
    plt.ylabel("PC2 - {0}%".format(per_var[1]))
    plt.title(f"{opt.mvl}PCA-sps{len(strName)}")
    pcaPlot.savefig(f"{PCAplotDir}/{opt.mvl}-PC1*PC2-{opt.pfx}-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}.pdf")

    ## PC1-PC3
    pcaPlot2 = plt.figure()
    plt.scatter(pca_df.PC1, pca_df.PC3,s=5)
    plt.xlabel("PC1 - {0}%".format(per_var[0]))
    plt.ylabel("PC3 - {0}%".format(per_var[1]))
    plt.title(f"{opt.mvl}PCA-sps{len(strName)}")
    pcaPlot2.savefig(f"{PCAplotDir}/{opt.mvl}-PC1*PC3-{opt.pfx}-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}.pdf")

    ## PC2-PC3
    pcaPlot3 = plt.figure()
    plt.scatter(pca_df.PC2, pca_df.PC3,s=5)
    plt.xlabel("PC1 - {0}%".format(per_var[0]))
    plt.ylabel("PC2 - {0}%".format(per_var[1]))
    plt.title(f"{opt.mvl}PCA-sps{len(strName)}")
    pcaPlot3.savefig(f"{PCAplotDir}/{opt.mvl}-PC2*PC3-{opt.pfx}-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}.pdf")

## Sub-functions in StrPCA, end
########################################



####################################################
#####################################################

def PCAcompAna(pca, paired_set, labels, PC_LC_Dir, gen_dict, opt):
    pcLC = pca.components_
    PCLinComb = pd.DataFrame(pcLC, columns= paired_set, index= labels)
    PCLinComb = PCLinComb.T #columns= labels , index= paired_set
    #print(PCLinComb)

    ### linear combination table
    PCLinComb.to_csv(f"{PC_LC_Dir}/{opt.mvl}-{opt.pfx}-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}.csv")

    ### BP Matrix Generation for PCs
    #PC1 = PCLinComb.PC1
    #pcMat = matrixGen(gen_dict, paired_set, PC1, opt)
    #plotBpMatrix(pcMat, PC_LC_Dir, opt)


def PCAcompAna_2(pca, paired_set, labels, PC_LC_Dir, opt):
    pcLC = pca.components_
    PCLinComb = pd.DataFrame(pcLC, columns= paired_set, index= labels)
    PCLinComb = PCLinComb.T
    if opt.mvl == "binary":
        PCLinComb.to_csv(f"{PC_LC_Dir}/binary-{opt.pfx}-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}.csv")
    elif opt.mvl == "mfe":
        PCLinComb.to_csv(f"{PC_LC_Dir}/mfe-{opt.pfx}-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}.csv")

#############################################
## Sub-functions in PCAcompAna, start

def matrixGen(gen_dict, bpset, DataArray, opt):
    sgLen = len(gen_dict[opt.seg])
    pcMat = np.zeros(shape = (sgLen,sgLen))
    for (bp, val) in zip(bpset, DataArray):
        assert isinstance(bp, tuple)
        #if val != 0: print(bp, val)
        (a,b) = bp
        pcMat[a,b] = val 
    #print(pcMat)
    #print(pcMat.shape)
    return pcMat

def plotBpMatrix(pcMat, PC_LC_Dir, opt):
    xmat = nan_to_num(pcMat, copy=True)
    if not opt.plm: plm = xmat.max()
    else:           plm = opt.plm
    if not opt.pli: pli = xmat.min()
    else:           pli = opt.pli
    if opt.pln: pcMat[pcMat <= opt.plt] = nan
    else: pcMat = xmat
    s = opt.pls
    n, m = pcMat.shape # default: 6.4, 4.8 -> 1.6 for legend
    mx = (4.8 / n) * m + 1.6
    pz = plt.figure(figsize=(mx*s,4.8*s))
    if opt.plc:
        colormap = ListedColormap(createColors(opt, "frac"), name='from_list', N=None) if opt.cls else 'viridis'
        plt.imshow(pcMat, cmap=colormap, vmin=pli, vmax=plm, interpolation="none")
    else:
        colormap = ListedColormap(createColors(opt, "frac")[::-1], name='from_list', N=None) if opt.cls else 'viridis_r'
        plt.imshow(pcMat, cmap=colormap, vmin=pli, vmax=plm, interpolation="none")
    ax = plt.gca()
    xs, xe = ax.get_xlim()
    ys, ye = ax.get_ylim()
    ax.xaxis.set_tick_params(rotation=45)
    ax.xaxis.set_ticks(arange(0, xe, opt.plx))
    ax.yaxis.set_ticks(arange(0, ys, opt.ply))
    plt.gcf().subplots_adjust(bottom=0.15*s)
    plt.xlabel(opt.seg)
    plt.ylabel(opt.seg)
    plt.colorbar()
    if opt.plv: pz.savefig(f"{PC_LC_Dir}/{opt.mvl}-{opt.pfx}-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}.svg", bbox_inches = 'tight', pad_inches = 0.1*s)
    pz.savefig(f"{PC_LC_Dir}/{opt.mvl}-{opt.pfx}-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}.svg", bbox_inches = 'tight', pad_inches = 0.1*s)
    plt.close(pz)

def createColors(opt, colortype): 
    ## add 48 colors
    collist = opt.cls.split(",")
    clln = len(collist)-1
    clr = int(round(48/clln,0))
    assert clln in [2,3,4,6,8,12,16,24,48]
    colorpalette = list()
    for i in range(clln):
        colorpalette.extend(getColorGradient(collist[i],collist[i+1],clr+int(bool(i)), colortype)[int(bool(i)):])
    return colorpalette

def getColorGradient(c1, c2, n, colortype):
    ## create color gradient from hex colors
    assert n > 1
    c1_rgb = array(hexToRGB(c1))/255
    c2_rgb = array(hexToRGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    outcolor = list()
    if colortype == "hex":    outcolor = ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]
    elif colortype == "frac": outcolor = [item for item in rgb_colors]
    elif colortype == "RGB":  outcolor = [[val*255 for val in item] for item in rgb_colors]
    else:                     outcolor = rgb_colors
    return outcolor


def hexToRGB(hex_str):
    ## #FFFFFF -> [255,255,255]
    # Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

## Sub-functions in PCAcompAna, end
#############################################










########################################################
########################################################

def umapClusterPostPCA_2D(PCAplotDir, UMAP_dir, opt):
    ## import PCA data 
    pca_df = pd.read_csv(f"{PCAplotDir}/{opt.mvl}PcaCoord-{opt.pfx}-{opt.seg}-sps{opt.sps}.csv", header=0, index_col=0) # dimensionality reduced by PCA to ~50

    ## perform UMAP to better separate clusters
    embedding = doUMAP_2D(pca_df, opt)

    ## identity clusters with DBSCAN
    clusLabel, clustered = doHDBSCAN(embedding, opt)

    ## evaluate clustering results using silhouette score
    S_score = metrics.silhouette_score(embedding ,clusLabel, metric="euclidean", random_state = 42)

    ## only UMAP projection
    plot_UMAP_projection(embedding, UMAP_dir, opt)

    ## plot results
    plot_2D_projection_UMAP_HDBSCAN(embedding, clusLabel, clustered, UMAP_dir, opt)

    return embedding, clusLabel, clustered, S_score


def umapClusterPostPCA_3D(PCAplotDir, UMAP_dir, opt):
    ## import PCA data 
    pca_df = pd.read_csv(f"{PCAplotDir}/{opt.mvl}PcaCoord-{opt.pfx}-{opt.seg}-sps{opt.sps}.csv", header=0, index_col=0) # dimensionality reduced by PCA to ~50

    ## perform UMAP to better separate clusters
    embedding = doUMAP_3D(pca_df, opt)

    ## identify clusters with DBSCAN
    clusLabel, clustered = doHDBSCAN(embedding, opt)

    ## evaluate clustering results using silhouette score
    try:
        S_score = metrics.silhouette_score(embedding ,clusLabel.labels_, metric="euclidean", random_state = 42)
    except:
        S_score = "NAN"

    ## Plot results
    plot_2D_projection_UMAP_HDBSCAN(embedding, clusLabel, clustered, UMAP_dir, opt)

    #plot_3D_UMAP_HDBSCAN(embedding, clusLabel, clustered, UMAP_dir, opt)
    #doMlab3D(embedding[:,0], embedding[:,1], embedding[:,2], 0.1)


    return embedding, clusLabel, clustered, S_score





###############################################################
## sub-functions in umapClusterPostPCA(), start

def doUMAP_3D(pca_df, opt):
    scaled_df = preprocessing.StandardScaler().fit_transform(pca_df)
    reducer = umap.UMAP(n_neighbors= opt.nnb , min_dist = opt.mdi, n_components = 3, random_state=42)
    embedding = reducer.fit_transform(scaled_df) # UMAP results, embedding is an np array
    return embedding


def doUMAP_2D(input_df, opt):
    scaled_df = preprocessing.StandardScaler().fit_transform(input_df)
    reducer = umap.UMAP(n_neighbors= opt.nnb , min_dist = opt.mdi, n_components = 2, random_state=42)
    embedding = reducer.fit_transform(scaled_df) # UMAP results, embedding is an np array
    return embedding


def doHDBSCAN(embedding, opt):
    clusLabel = hdbscan.HDBSCAN(min_samples = opt.msp, min_cluster_size = opt.mcs).fit(embedding)
    #clustered = (clusLabel >= 0)
    clustered = (clusLabel.labels_ >= 0)
    return clusLabel, clustered


def plot_UMAP_projection(embedding, UMAP_dir, opt):
    umapPlot = plt.figure()
    plt.scatter(embedding[:,0], embedding[:,1], s=0.2)
    plt.title(f"UMAP-numNeighbor{opt.nnb}-minDist{opt.mdi}")
    umapPlot.savefig(f"{UMAP_dir}/UMAP-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}-numNeighbor{opt.nnb}-minDist{opt.mdi}.pdf")


def plot_2D_projection_UMAP_HDBSCAN(embedding, clusLabel, clustered, UMAP_dir, opt):
    umapPlot = plt.figure()
    color_palette = sns.color_palette("Paired", len(clustered))
    cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusLabel.labels_]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                         zip(cluster_colors, clusLabel.probabilities_)]
    plt.scatter(embedding[:,0], embedding[:,1], s=5, c=cluster_member_colors, alpha = 0.25)
    plt.title(f"UMAP_HDBSCAN-{opt.mvl}-numNeighbor{opt.nnb}-minDist{opt.mdi}-minSamp{opt.msp}-minCluSize{opt.mcs}")
    umapPlot.savefig(f"{UMAP_dir}/UMAP-HDBSCAN-{opt.mvl}-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}-numNeighbor{opt.nnb}-minDist{opt.mdi}-minSamp{opt.msp}-minCluSize{opt.mcs}.pdf")


def plot_3D_UMAP_HDBSCAN(embedding, clusLabel, clustered, UMAP_dir, opt):
    umapPlot = plt.figure()
    ax = Axes3D(umapPlot)
    ax = umapPlot.add_subplot(111, projection="3d")
    color_palette = sns.color_palette("Paired", len(clustered))
    cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusLabel.labels_]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                         zip(cluster_colors, clusLabel.probabilities_)]
    ax.scatter(embedding[:,0], embedding[:,1],embedding[:,2], s=5, c=cluster_member_colors, alpha = 0.25)
    ax.set_title(f"UMAP_HDBSCAN-{opt.mvl}-numNeighbor{opt.nnb}-minDist{opt.mdi}-minSamp{opt.msp}-minCluSize{opt.mcs}")
    umapPlot.savefig(f"{UMAP_dir}/3D_UMAP-HDBSCAN-{opt.mvl}-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}-numNeighbor{opt.nnb}-minDist{opt.mdi}-minSamp{opt.msp}-minCluSize{opt.mcs}.pdf")

def doMlab3D(x, y, z, scaling_factor):
    mlab.figure()
    mlab.points3d(x, y, z, scale_factor=scaling_factor)
    mlab.show()


## sub-functions in umapClusterPostPCA(), end
#######################################################








#############################################################
#############################################################

def strExtract(pfx_dir, gen_dict, StrBP_dir, clusLabel,clustered, opt):
    StrBP_df = pd.read_csv(f"{StrBP_dir}/{opt.mvl}-{opt.pfx}-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}.csv", header=0, index_col=0) #df = pd.DataFrame(StrListlist, columns= paired_set, index= strName)
    strList = StrBP_df.index
    allbp = StrBP_df.columns 
    #print(StrBP_df)
    cluster_matrix_dir = os.path.join(pfx_dir,"ClusterStructure", f"{opt.mvl}-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}-numNeighbor{opt.nnb}-minDist{opt.mdi}-minSamp{opt.msp}-minCluSize{opt.mcs}")
    if not os.path.isdir(cluster_matrix_dir):
        os.makedirs(cluster_matrix_dir, 0o777)

    ### extract str names in each cluster, dict sorted by keys (cluster name)
    clusteredStr_dict,n = strNameExtract(clusLabel, strList, opt)

    ### extract bp info in each cluster , dict sorted by keys (cluster name)
    clusterDF_dict = dfExtract(clusteredStr_dict, StrBP_df, allbp)

    clusMat_dict, clusbpdict_dict = {}, {}
    ### Plot matrix for each cluster, output matrix dict {cluster n}:matrix, output sbp_dict
    for i, key in enumerate(clusterDF_dict.keys()):
        cluster_df = clusterDF_dict[key]
        bpset = cluster_df.columns
        bpWeight = cluster_df.loc['Total']
        clusStrMat = matrixGen_2(gen_dict, bpset, bpWeight, opt)
        clusMat_dict[key] = clusStrMat
        sbp_dict = sbpdict_Gen(bpset, bpWeight, opt)
        clusbpdict_dict[key] = sbp_dict
        plotClusterMatrix(i, clusStrMat, cluster_matrix_dir, opt)

    plot_condensed_tree(clusLabel, clustered, cluster_matrix_dir)

    return clusMat_dict, clusbpdict_dict, clusteredStr_dict, n



def strExtract_2(pfx_dir, gen_dict, StrBP_dir, clusLabel, paired_set, opt):
    StrBP_df = pd.read_csv(f"{StrBP_dir}/{opt.mvl}-{opt.pfx}-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}.csv", header=0, index_col=0) #df = pd.DataFrame(StrListlist, columns= paired_set, index= strName)
    strList = StrBP_df.index
    #print(StrBP_df)
    cluster_matrix_dir = os.path.join(pfx_dir,"ClusterStructure", f"{opt.mvl}-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}-numNeighbor{opt.nnb}-minDist{opt.mdi}-minSamp{opt.msp}-minCluSize{opt.mcs}")
    if not os.path.isdir(cluster_matrix_dir):
        os.makedirs(cluster_matrix_dir, 0o777)

    ### extract str names in each cluster
    clusteredStr_dict = strNameExtract(clusLabel, strList, opt)
    #print(clusteredStr_dict)

    ### extract bp info in each cluster, store as dictionary
    clusterDF_dict = dfExtract_2(clusteredStr_dict, StrBP_df, paired_set)

    ### Plot matrix for each cluster
    mat_list = []
    for i, key in enumerate(clusterDF_dict.keys()):
        cluster_df = clusterDF_dict[key] # df for individual s
        bpWeight = cluster_df.loc['Total']
        clusStrMat = matrixGen(gen_dict, paired_set, bpWeight, opt)
        #print(clusStrMat[clusStrMat != 0])
        mat_list.append(clusStrMat)
        plotClusterMatrix(i, clusStrMat, cluster_matrix_dir, opt)
        #print(type(clusStrMat))
    return mat_list

#######################################################
## sub-functions in strExtract(), start

def strNameExtract(clusLabel, strList, opt):
    clusteredStr_dict = {}
    n = 0
    for i, cluster in enumerate(clusLabel.labels_):
        if cluster < 0: continue
        n += 1
        strVal = clusteredStr_dict.get(cluster,[])
        strVal.append(strList[i])
        clusteredStr_dict[cluster] = strVal
    clusteredStr_dict = dict(sorted(clusteredStr_dict.items(), key=itemgetter(0)))
    print(f"Total point = {opt.sps}, {n} points were included clustered")
    return clusteredStr_dict, n


def dfExtract(clusteredStr_dict, StrBP_df, paired_set):
    clusterDF_dict = {}
    for i, key in enumerate(clusteredStr_dict.keys()):
        print(f"Cluster {key} contains {len(clusteredStr_dict[key])} points")
        strBP_list = []
        for structure in clusteredStr_dict[key]:
            strBP = StrBP_df.loc[str(structure)]
            #print(strBP)
            strBP_list.append(strBP)
        #print(strBP_list)
        clusterDF = pd.DataFrame(strBP_list, columns=paired_set, index= clusteredStr_dict[key])
        clusterDF.loc['Total']= clusterDF.sum()
        #print(clusterDF)
        clusterDF_dict[key] = clusterDF
    
    return clusterDF_dict


def dfExtract_2(clusteredStr_dict, StrBP_df, paired_set):
    clusterDF_dict = {}
    print(StrBP_df.head(10))
    for i, (key,row_index) in enumerate(clusteredStr_dict.items()):
        print(f"Cluster {key} contains {len(clusteredStr_dict[key])} points")
        clusterDF = StrBP_df.loc[row_index]
        clusterDF.loc['Total'] = clusterDF.sum()
        print(clusterDF)
        clusterDF_dict[key] = clusterDF
    return clusterDF_dict

def matrixGen_2(gen_dict, bpset, DataArray, opt):
    sgLen = len(gen_dict[opt.seg])
    pcMat = np.zeros(shape = (sgLen,sgLen))
    for (bp, val) in zip(bpset, DataArray):
        res = ast.literal_eval(bp)
        (a,b) = res
        #print(a)
        #print(b)
        #row = bp
        #print(bp)
        #print(f"({j},{i})")
        #print(PCLinComb.iloc[j, i])
        pcMat[a,b] = val #data_point = df.iloc[row_index, column_index]
    #print(pcMat)
    #print(pcMat.shape)
    return pcMat


def sbpdict_Gen(bpset, bpWeight, opt):
    sbp_dict = {}
    for (bp, val) in zip(bpset, bpWeight):
        res = ast.literal_eval(bp)
        (a,b) = res
        sbp_dict[(a,b)] = val
    sbp_dict = dict(sorted(sbp_dict.items(), key = itemgetter(1),reverse=True))
    #print(sbp_dict)
    return sbp_dict
 
def plotClusterMatrix(i, pcMat, cluster_matrix_dir, opt):
    xmat = nan_to_num(pcMat, copy=True)
    if not opt.plm: plm = xmat.max()
    else:           plm = opt.plm
    if not opt.pli: pli = xmat.min()
    else:           pli = opt.pli
    if opt.pln: pcMat[pcMat <= opt.plt] = nan
    else: pcMat = xmat
    s = opt.pls
    n, m = pcMat.shape # default: 6.4, 4.8 -> 1.6 for legend
    mx = (4.8 / n) * m + 1.6
    pz = plt.figure(figsize=(mx*s,4.8*s))
    if opt.plc:
        colormap = ListedColormap(createColors(opt, "frac"), name='from_list', N=None) if opt.cls else 'viridis'
        plt.imshow(pcMat, cmap=colormap, vmin=pli, vmax=plm, interpolation="none")
    else:
        colormap = ListedColormap(createColors(opt, "frac")[::-1], name='from_list', N=None) if opt.cls else 'viridis_r'
        plt.imshow(pcMat, cmap=colormap, vmin=pli, vmax=plm, interpolation="none")
    ax = plt.gca()
    xs, xe = ax.get_xlim()
    ys, ye = ax.get_ylim()
    ax.xaxis.set_tick_params(rotation=45)
    ax.xaxis.set_ticks(arange(0, xe, opt.plx))
    ax.yaxis.set_ticks(arange(0, ys, opt.ply))
    plt.gcf().subplots_adjust(bottom=0.15*s)
    plt.xlabel(opt.seg)
    plt.ylabel(opt.seg)
    plt.colorbar()
    pz.savefig(f"{cluster_matrix_dir}/Cluster{i}-s921-d921_from_start.pdf", bbox_inches = 'tight', pad_inches = 0.1*s)
    plt.close(pz)

def plot_condensed_tree(clusLabel, clustered, cluster_matrix_dir):
    clusLabel.condensed_tree_.plot(select_clusters=True, label_clusters = True ,  selection_palette = sns.color_palette("Paired", len(clustered)))
    #hdbscan.plots.CondensedTree(clusLabel.condensed_tree_)
    plt.savefig(f"{cluster_matrix_dir}/condensed_tree.pdf")

## sub-functions in strExtract(), end
########################################################






#########################################################
#########################################################

def clusEvaluate(gen_dict, mat_dict, pfx_dir, S_score, clusteredStr_dict, n, opt):
    sgLen = len(gen_dict[opt.seg])
    numCluster = len(list(mat_dict.keys()))
    clusInfo, i = f"", 0
    while i < numCluster:
        clusInfo = clusInfo + f"Cluster {list(clusteredStr_dict.keys())[i]} contains {len(list(clusteredStr_dict.values())[i])} points; "
        i += 1

    #print(sgLen)

    #pair_diff = cal_pairwiseDiff(mat_list, sgLen)
    diff_from_ave = cal_diff_from_ave(mat_dict, sgLen)

    #print(f"S-score: {S_score}, pair diff: {pair_diff}, diff_from_ave: {diff_from_ave}")
    print(f"S-score: {S_score}, diff_from_ave: {diff_from_ave}")

    sf = f"{pfx_dir}/score_file_{opt.pfx}-{opt.mvl}-{opt.seg}-sps{opt.sps}-ncp{opt.ncp}.tsv"
    writeheader = False if os.path.exists(sf) else True
    with open(sf, "a") as sft:
        if writeheader: 
            sft.write(f"matrixVal\tsegment\tsampleSize\tNumComponents\tnNeighbors\tminDistance\tminSamp\tminClusterSize\tPointsClustered \tnumCluster\tClusterInfo\tSilhouetteScore\tDiff_From_Aver\n")
            sft.write(f"{opt.mvl}\t{opt.seg}\t{opt.sps}\t{opt.ncp}\t{opt.nnb}\t{opt.mdi}\t{opt.msp}\t{opt.mcs}\t{n}\t{numCluster}\t{clusInfo}\t{S_score}\t{diff_from_ave}\n")
        else:
            sft.write(f"{opt.mvl}\t{opt.seg}\t{opt.sps}\t{opt.ncp}\t{opt.nnb}\t{opt.mdi}\t{opt.msp}\t{opt.mcs}\t{n}\t{numCluster}\t{clusInfo}\t{S_score}\t{diff_from_ave}\n")





################################################
## sub-functions in clusEvaluate(), start

def cal_pairwiseDiff(mat_dict, sgLen):
    sum_pairwise_diff = 0
    sum_pairwise_diff_list = []
    norm_mat_list = []
    num_cluster = len(list(mat_dict.keys()))
    num_combs = math.comb(num_cluster, 2)
    for mat in mat_dict.values():
        #print(mat[mat != 0])
        norm_mat = normalise_mat(mat, sgLen)
        #print(norm_mat[norm_mat!=0])
        norm_mat_list.append(norm_mat)
        #print(mat.shape)
        #print(f"{mat}, before normalisation")
        #print(f"{norm_mat}, after normalisation")
    comb = combinations(norm_mat_list,2)
    for (mat1,mat2) in list(comb):
        for i, (row1, row2) in enumerate(zip(mat1,mat2)):
            for j, (ele1, ele2) in enumerate(zip(row1,row2)):
                per_diff_pair = np.nan_to_num(abs(ele1-ele2)/((ele1+ele2)*0.5))
                sum_pairwise_diff += per_diff_pair
                sum_pairwise_diff_list.append(per_diff_pair)
                #print(f"for {mat1}, \n{mat2}, \n {ele1}-{ele2} {i}-{j} per_diff is {per_diff_pair}")
    pair_diff = sum_pairwise_diff/num_combs
    non_zero = [x for x in sum_pairwise_diff_list if x!=0]
    #print(non_zero)
    print(pair_diff)
    return pair_diff



def cal_diff_from_ave(mat_dict, sgLen):
    sum_of_diff = 0
    num_cluster = len(list(mat_dict.keys()))
    aver_mat = find_aver_mat(mat_dict, sgLen)
    normed_aver_mat = normalise_mat(aver_mat, sgLen)
    #print(f"normed_aver_mat is {normed_aver_mat}")
    for mat in mat_dict.values():
        norm_mat = normalise_mat(mat, sgLen)
        #print(norm_mat)
        for i, row in enumerate(norm_mat):
            for j, element in enumerate(row):
                per_diff = np.nan_to_num(abs(element - normed_aver_mat[i,j])/normed_aver_mat[i,j])
                #print(f"per_diff for {i},{j} is: {per_diff}")
                sum_of_diff += per_diff

    diff_from_ave = sum_of_diff/num_cluster
    print(sum_of_diff)
    return diff_from_ave


def normalise_mat(in_mat, sgLen):
    out_mat = np.zeros(shape = (sgLen, sgLen))
    largest_val = np.amax(in_mat)
    #print(f"in in_mat {in_mat}, the largest value is {largest_val}")
    for i, row in enumerate(in_mat):
        for j, element in enumerate(row):
            #print(f"{i}-{j}, element: {element}, largest val: {largest_val}")
            new_val = element/largest_val
            out_mat[i,j] = new_val
    #print(f"normed_mat is {out_mat}")
    out_mat = np.nan_to_num(out_mat)
    return out_mat



def find_aver_mat(mat_dict, sgLen):
    num_cluster = len(list(mat_dict.keys()))
    aver_mat = np.zeros(shape = (sgLen,sgLen))
    for mat in mat_dict.values():
        for i, row in enumerate(mat):
            for j, element in enumerate(row):
                aver_mat[i,j] += mat[i,j]
    aver_mat = aver_mat/num_cluster
    return aver_mat

## sub-functions in clusEvaluate(), end
################################################











###########################################################
###########################################################


def matrix_to_jc(gen_dict, clusbpdict_dict, opt):
    #print(list(clusbpdict_dict.values())[0])
    clusJc_dict = {}
    for cluster, sbp_dict in clusbpdict_dict.items():
        #print(f"for cluster{cluster}, sbp_dict: {sbp_dict}")
        mate_dict = getMates(sbp_dict)
        nmate_dict = combineMates(opt, mate_dict)
        #print(sbp_dict)
        #print(f" nmate_dict:{nmate_dict}")
        jclist = mateToJunctions(opt, nmate_dict, gen_dict, opt.seg)
        report_list = [(jc.RNA, jc.pattern) for jc in jclist if not re.search(r"[(]|[)]", jc.pattern)]
        print(f"for cluster {cluster}, report_list:\n{report_list}")
        clusJc_dict[cluster] = jclist
    return clusJc_dict

####################################################
## sub-functions in mat_to_jc(), start


def getMates(sbp_dict):
    ## extract junction mates
    mate_dict = dict()
    while sbp_dict:
        (i,j) = list(sbp_dict.keys())[0]
        pbc = sbp_dict.pop((i,j))
        if not pbc: continue
        read_list = [(i,j,pbc)]
        pbcl, pbcr = pbc, pbc
        s = 0
        while pbcl or pbcr:
            s += 1
            if pbcl: pbcl = sbp_dict.pop((i+s,j-s), 0)
            if pbcr: pbcr = sbp_dict.pop((i-s,j+s), 0)
            if pbcl: read_list.append((i+s,j-s,pbcl))
            if pbcr: read_list.append((i-s,j+s,pbcr))
        ilist, jlist, clist = zip(*sorted(read_list))
        mate_dict[(min(ilist),max(ilist)+1,min(jlist),max(jlist)+1)] = list(clist)
        # = {(ai, aj, bi, bj): val}
    return mate_dict

def combineMates(opt, mate_dict): 
    #+ combine mates
    checked = set()
    nmate_dict = dict()
    for (lai,laj,lbi,lbj),llist in mate_dict.items():
        if (lai,laj,lbi,lbj) in checked: continue
        mate_list = list()
        for (ai,aj,bi,bj),clist in mate_dict.items():
            if (ai,aj,bi,bj) in checked: continue
            elif (lai,laj,lbi,lbj) == (ai,aj,bi,bj):
                mate_list.append((ai,aj,bi,bj,clist))
                checked.add((ai,aj,bi,bj))
            elif lai-opt.jcg <= aj <= lai and lbj <= bi <= lbj+opt.jcg:
                if np.mean(clist) <= np.mean(llist)*opt.jcc: continue
                mate_list.insert(0, (ai,aj,bi,bj,clist))
                checked.add((ai,aj,bi,bj))
            elif laj <= ai <= laj+opt.jcg and lbi-opt.jcg <= bj <= lbi:
                if np.mean(clist) <= np.mean(llist)*opt.jcc: continue
                mate_list.append((ai,aj,bi,bj,clist))
                checked.add((ai,aj,bi,bj))
        ail, ajl, bil, bjl, clist= zip(*mate_list)
        clist = [c for cl in clist for c in cl]
        nmate_dict[(min(ail),max(ajl),min(bil),max(bjl))] = clist
    return nmate_dict

def mateToJunctions(opt, nmate_dict, gen_dict, Seg): 
    ## transform mates to junctions
    jclist = list()
    Len = len(gen_dict[Seg])
    for i, ((ai,aj,bi,bj), clist) in enumerate(nmate_dict.items()):
        ai, aj, bi, bj = max(0,ai), min(aj,Len), max(0, bi), min(bj, Len)
        if aj >= bi-2:
            ai, aj, bi, bj = ai, bj, -1, -1
            RNA = gen_dict[Seg][ai:aj]
        else: RNA = gen_dict[Seg][ai:aj] + "&" + gen_dict[Seg][bi:bj]
        mfe, pattern = doCofold_2(RNA, opt)
        chjc = {"Seg":Seg, "ai":ai, "aj":aj, "bi":bi, "bj":bj, "reverse":False, "Weight":np.mean(clist), "jPeak":max(clist), "idx":i, "pattern":pattern, "mfe":mfe, "RNA": RNA}
        jc = junction(**chjc)
        jclist.append(jc)
        jclist = sorted(jclist, key = attrgetter("Weight"), reverse= True)
        #Weight_list = [jc.Weight for jc in jclist]
        #print(Weight_list)
        #pattern_list = [jc.pattern for jc in jclist]
        #print(pattern_list)

    return jclist


## sub-functions in mat_to_jc(), end
####################################################





























##################
## parser
################################################################################

class options(object):
    def __init__(self, **data):
        self.__dict__.update((k,v) for k,v in data.items())
    def plot(self, sep):
        ldat = sep.join([f"{var}" for key,var in vars(self).items()])
        return ldat

if __name__ == "__main__":

    ############################################################################
    ## get time and save call
    sscript = sys.argv[0]
    start_time = time.time()
    current_time = time.strftime('%x %X')
    scall = " ".join(sys.argv[1:])
    with open(f"{sscript}.log", "a") as calllog:
        calllog.write(f"Start : {current_time}\n")
        calllog.write(f"Script: {sscript}\n")
        calllog.write(f"Call  : {scall}\n")
    print(f"Call: {scall}")
    print(f"Status: Started at {current_time}")
    ############################################################################
    ## transform string into int, float, bool if possible
    def trans(s):
        if isinstance(s, str):
            try: return int(s)
            except ValueError:
                try: return float(s)
                except ValueError:
                    if s in ["True", "False"]: return s == "True"
                    else: return s
        else: return s
    ############################################################################
    ## save documentation
    rx_text = re.compile(r"\n^(.+?)\n((?:.+\n)+)",re.MULTILINE)
    rx_oneline = re.compile(r"\n+")
    rx_options = re.compile(r"\((.+?)\:(.+?)\)")
    help_dict, type_dict, text_dict, mand_list = {}, {}, {}, []
    for match in rx_text.finditer(script_usage):
        argument = match.groups()[0].strip()
        text = " ".join(rx_oneline.sub("",match.groups()[1].strip()).split())
        argopts = {"action":"store", "help":None, "default":None, "choices":None}
        for option in rx_options.finditer(text):
            key = option.group(1).strip()
            var = option.group(2).strip()
            if var == "False": argopts["action"] = "store_true"
            if var == "True": argopts["action"] = "store_false"
            if key == "choices": var = [vs.strip() for vs in var.split(",")]
            if key == "default": var = trans(var)
            argopts[key] = var
        if argopts["default"]: add_default = f" (default: {str(argopts['default'])})"
        else: add_default = ""
        argopts["help"] = rx_options.sub("",text).strip()+add_default
        argnames = argument.split(",")
        if len(argnames) > 1:
            if argopts["default"] == None:
                mand_list.append(f"{argnames[1][1:]}")
            type_dict[f"{argnames[1][1:]}"] = argopts["default"]
            argopts["argshort"] = argnames[1]
            help_dict[argnames[0]] = argopts
        else:
            text_dict[argnames[0]] = argopts["help"]
    ############################################################################
    ## get arguments
    if text_dict["dependencies"]:
        desc = f"{text_dict['description']} (dependencies: {text_dict['dependencies']})"
    else:
        desc = text_dict['description']
    p = ap.ArgumentParser(prog=sscript, prefix_chars="-", usage=text_dict["usage"],
                          description=desc, epilog=text_dict["reference"])
    p.add_argument("-v", "--version", action="version", version=text_dict["version"])
    for argname,argopts in help_dict.items():
        argshort = argopts["argshort"]
        if argopts["choices"]:
            p.add_argument(argshort, argname,            dest=f"{argshort[1:]}",\
                           action=argopts["action"],     help=argopts["help"],\
                           default=argopts["default"],   choices=argopts["choices"])
        else:
            p.add_argument(argopts["argshort"], argname, dest=f"{argshort[1:]}",\
                           action=argopts["action"],     help=argopts["help"],\
                           default=argopts["default"])
    p._optionals.title = "arguments"
    opt = vars(p.parse_args())
    ############################################################################
    ## validate arguments
    if None in [opt[mand] for mand in mand_list]:
        print("Error: Mandatory arguments missing!")
        print(f"Usage: {text_dict['usage']} use -h or --help for more information.")
        sys.exit()
    for key,var in opt.items():
        if key not in mand_list:
            arg_req, arg_in = type_dict[key], trans(var)
            if type(arg_req) == type(arg_in):
                opt[key] = arg_in
            else:
                print(f"Error: Argument {key} is not of type {type(arg_req)}!")
                sys.exit()
    ############################################################################
    ## add log create options class
    opt["log"] = True
    copt = options(**opt)
    ############################################################################
    ## call main function
    try:
        #saved = main(opt)
        saved = main(copt)
    except KeyboardInterrupt:
        print("Error: Interrupted by user!")
        sys.exit()
    except SystemExit:
        print("Error: System exit!")
        sys.exit()
    except Exception:
        print("Error: Script exception!")
        traceback.print_exc(file=sys.stderr)
        sys.exit()
    ############################################################################
    ## finish
    started_time = current_time
    elapsed_time = time.time()-start_time
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    current_time = time.strftime('%x %X')
    if saved:
        with open(f"{sscript}.log", "a") as calllog,\
             open(os.path.join(saved,f"call.log"), "a") as dirlog:
            calllog.write(f"Save  : {os.path.abspath(saved)}\n")
            calllog.write(f"Finish: {current_time} in {elapsed_time}\n")
            ## dirlog
            dirlog.write(f"Start : {started_time}\n")
            dirlog.write(f"Script: {sscript}\n")
            dirlog.write(f"Call  : {scall}\n")
            dirlog.write(f"Save  : {os.path.abspath(saved)}\n")
            dirlog.write(f"Finish: {current_time} in {elapsed_time}\n")
    print(f"Status: Saved at {saved}")
    print(f"Status: Finished at {current_time} in {elapsed_time}")
    sys.exit(0)
