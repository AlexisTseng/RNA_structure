#!/usr/bin/env python3
# script: foldSPS.py
# author: Daniel Desiro', Jiayi Zeng
script_usage="""
usage
    foldSPS.py -fsa <fasta_genome> -pfx <out_prefix> -psp <SPLASH-table> [options]

version
    foldSPS.py 0.0.1 (alpha)

dependencies
    Python v3.9.7, Scipy v1.8.0, NumPy v1.22.2, Matplotlib v3.5.1

description
    bla

################################################################

--prefix,-pfx
    output directory and prefix for result files

--genomeFasta,-fsa
    input genome FASTA file

--parseSPLASHintra,-psp
    input intra-molecular interaction table from parseSPLASH

--parseSPLASHinter,-psi
    input inter-molecular interaction table from parseSPLASH (default: )

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
    (default: #ae0001,#222f5b,#2a623d,#f0c75e)

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
import os
import sys
import re
import pickle
import argparse as ap
import time
from numpy import zeros, nan_to_num, arange, nan, ones, mean, median, std, cov, array, quantile, log10, log2, linspace
from operator import attrgetter, itemgetter

try:
    from RNA import fold_compound, cvar, CONSTRAINT_DB, CONSTRAINT_DB_DEFAULT, bp_distance
except:
    print("Error: The mandatory library ViennaRNA 2.5 is missing, please read the README.md for more information!")
    exit()





##################################################################
## Main function
##################################################################

def main(opt):
    ####################################################
    ## Step1: make output directory
    pfx_dir = makeDir(opt)
    
    #############################################
    ## Step 2：import genome fasta file
    gen_dict = readGenome(opt)
    #print(gen_dict)
    ###############################################
    ## Step 3: mport SPLASH output tsv table
    junction_list = list()
    segjc_dict = readSPLASHtable(opt, junction_list, gen_dict)
    #print(junction_list[:5])
    #print(all_s1)
    ######################################################################
    ## Step 4 Version 1: loop through jc, generate constraint string
    #constr_dict, constr_SRI_dict, constr_LRI_dict = genConstraintDictIntra(gen_dict,segjc_dict,opt)
    ##################################################################
    ## Step 4 Version 2: generate a list of non-overlapping constraint strings
    constr_dict, bpdict = getConsSet(gen_dict,segjc_dict,opt)

    #print(constr_dict["S10"][0])
    #print(bpdict["S10"][0])
    #print(gen_dict["S10"])


    ########################################################################
    # Step 5: Generate RNA structure
    #genRNAstr(constr_dict,gen_dict,opt)


def genRNAstr(constr_dict,gen_dict,opt):
    for kg, seq in gen_dict.items():
        for i, str in enumerate(constr_dict[kg]):
            mfe, pattern = doCofold(seq, str, opt)
            print(f"{kg}-{i+1}:\n mfe: {mfe} \n seq: {seq} \n pattern:{pattern} \n")



###################################################################
## Operations in Main Function
###################################################################

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
    

def getConsSet(gen_dict,segjc_dict,opt):
    constr_dict, bpdict = initConsBplog_2(gen_dict) #value = list of list
    for (kc,vc),vb in zip(constr_dict.items(), bpdict.values()):
        vs = segjc_dict[kc]
        for k,jc in enumerate(vs):
            #if k > 30:
            #    exit()
            print(k, jc.aSeq, jc.ai, jc.bi, jc.pattern, jc.jType)
            bplist = sorted(getBasePairs(jc.ai, jc.bi, jc.pattern))
            for i, (str,bplog) in enumerate(zip(constr_dict[kc], bpdict[kc])):#loop through constraint values and bp value (list of list) tgt
                test_str, allp = getTestStr(bplist, str)
                #if i >= 2 and jc.jCount < 100:
                #    print("discarded")
                #    break
                if isOverlap(test_str) or isPknot_2(bplist,str,bplog): # either overlap or pknot
                    if i+1 != len(vc) : #not in the last element, continue to the next 
                        continue
                    else: # not in the last element, create new and update
                        constr_dict, bpdict = newStrBlog(kc, constr_dict, bpdict)
                        #constr_dict, bpdict = updateStrBpDict(bplist,i+1,kc,constr_dict, bpdict)
                        print(f"Create new string.")
                else: #neither overlap or knot in the present str, update
                    constr_dict, bpdict = updateStrBpDict(bplist,i,kc,constr_dict, bpdict)
                    print(f"Update string. \n{kc}-{i+1}:{list_to_str(constr_dict[kc][i])}")
                    break
        #print(f"For {kc}: {len(constr_dict[kc])} strings")
        constr_dict = dictValueToStr(constr_dict,kc)
    constrSetOverview(constr_dict)
    return constr_dict, bpdict

def constrSetOverview(constr_dict):
    for kc,vc in constr_dict.items():
        print(f"For {kc}, {len(constr_dict[kc])} strings present")
        for item in vc:
            print(f"{item} \n")


def isOverlap(test_str):
    ss = 0
    if re.search(r"[(]|[)]",test_str) == None:
        print("no overlap")
    else:
        print(f"Overlap")
        ss += 1
    return bool(ss)


def updateStrBp(bplist, constr_list, bplog):
    for (p1,p2) in bplist:
        constr_list[p1] = "("
        constr_list[p2] = ")"
        bplog[p1] = p2
        bplog[p2] = p1
    return constr_list, bplog

def updateStrBpDict(bplist,pos,seg,constr_dict, bpdict):
    for (p1,p2) in bplist:
        constr_dict[seg][pos][p1] = "("
        constr_dict[seg][pos][p2] = ")"
        bpdict[seg][pos][p1] = p2
        bpdict[seg][pos][p2] = p1
    return constr_dict, bpdict
    
def dictValueToStr(dict,seg):
    for i, item in enumerate(dict[seg]):
        dict[seg][i] = list_to_str(item)
    return dict


def newStrBlog(seg, constr_dict, bpdict):
    length, str_init, bp_init = len(constr_dict[seg][0]), ".", -1
    constr_dict[seg].append([str_init]*length)
    bpdict[seg].append([bp_init]*length)
    return constr_dict, bpdict


def initConsBplog_2(gen_dict):
    ### generate empty constraint string
    constr_dict, const = {},"."
    ### generate empty base paring log dict
    bpdict, init = {}, -1
    ### Initialise constraints and bplog
    for k, v in gen_dict.items():
        constr_dict[k] = []
        constr_dict[k].append([const]*len(v)) # operate as lists, then report as strings
        bpdict[k] = []
        bpdict[k].append([init]*len(v))
    return constr_dict, bpdict

def isPknot_2(bplist,constr_list,bplog):
    e1,s2 = bplist[-1][0], bplist[-1][-1]
    ss = 0
    if [i for i,j in enumerate(bplog[e1+1:s2]) if j != -1]: #bp present
        if e1 < min([j for j in bplog[e1+1:s2] if j != -1]) and s2 > max([j for j in bplog[e1+1:s2] if j != -1]):
            #print(f"Not pseudo knot: {list_to_str(constr_list[e1:s2+1])}")
            print(f"Not pseudo knot")
            pass
        else:
            #print(f"Is pseudo knot: {list_to_str(constr_list[e1:s2+1])}")
            print(f"is pseudo knot")
            ss += 1
    else: # bp not present
        #print(f"Not pseudo knot: {list_to_str(constr_list[e1:s2+1])}")
        print(f"Not pseudo knot")
    return bool(ss)


def knotUpdate(seg, type, bplist, bpdict_SRI, bpdict_LRI, vSRI, vLRI):
    discard, out_type = 0, ""
    if type == "SRI":
        if isPknot(seg, bplist, bpdict_SRI, vSRI): #knot in SRI
            if isPknot(seg, bplist, bpdict_LRI, vLRI): # knot in LRI
                discard += 1
            else: #not knot in LRI
                out_type = "LRI"
        else: # not knot in SRI
            out_type = "SRI"
    elif type == "LRI":
        if isPknot(seg, bplist, bpdict_LRI, vLRI):
            discard += 1
        else:
            out_type = "LRI"
    else:
        print("jc.jType not SRI or LRI")
    return discard, out_type

#####################################################################################


def genConstraintDictIntra(gen_dict,segjc_dict,opt):
    n_ovp, n_total = 0, 0
    constr_dict, constr_SRI_dict, constr_LRI_dict, bpdict_SRI, bpdict_LRI = initConsBplog(gen_dict)
    ### update constraint string using jc dictionary
    for kc,vc in constr_dict.items():
        vs, vSRI, vLRI = segjc_dict[kc], constr_SRI_dict[kc], constr_LRI_dict[kc]
        for i,jc in enumerate(vs):
            n_total += 1
            print(jc.aSeq, i, jc.ai, jc.bi, jc.pattern, jc.jType)
            bplist = sorted(getBasePairs(jc.ai, jc.bi, jc.pattern))
            jc.bplist = bplist
            test_str, allp = getTestStr(bplist, vc)
            #print(test_str) # all element in the constraint string at the position of "(" and ")"
            if re.search(r"[(]|[)]|[x]",test_str) == None:
                discard, out_type  = knotUpdate(kc, jc.jType, bplist, bpdict_SRI, bpdict_LRI, vSRI, vLRI)
                if discard == 1:
                    continue
                else:
                    vc, vSRI, vLRI = constrUpdate(out_type, vc, vSRI, vLRI, bplist, allp)
                    bpdict_SRI, bpdict_LRI = bpUpdate(kc,out_type, bplist, bpdict_SRI, bpdict_LRI)
            else: # there is overlap
                n_ovp +=1
                getConstrOverlap(jc.pattern,vc,jc.ai,jc.bi)
                continue
            constr_dict, constr_SRI_dict, constr_LRI_dict = constr_dictUpdate(kc, constr_dict, constr_SRI_dict, constr_LRI_dict, vc, vSRI, vLRI) #update constraint list after each jc object
            #print(f"{jc.aSeq}-{i}: {list_to_str(vc)} \nbpSRI: {bpdict_SRI[kc]}\nbpLRI:{bpdict_LRI[kc]}\nEND")
            if i > 10:
                exit()
        constr_dict[kc],constr_SRI_dict[kc], constr_LRI_dict[kc] = list_to_str(vc), list_to_str(vSRI), list_to_str(vLRI) # final conversion of list into string
    #print(f"Total jc: {n_total}\noverlap: {n_ovp}\n{constr_dict}")
    return constr_dict, constr_SRI_dict, constr_LRI_dict





############################################################################
## 
def initConsBplog(gen_dict):
    ### generate empty constraint string
    constr_dict, constr_SRI_dict, constr_LRI_dict, const = {},{}, {}, "."
    ### generate empty base paring log dict
    bpdict_SRI, bpdict_LRI, init = {}, {}, -1
    ### Initialise constraints and bplog
    for k, v in gen_dict.items():
        constr_dict[k] = [const]*len(v)
        constr_LRI_dict[k] = [const]*len(v)
        constr_SRI_dict[k] = [const]*len(v) # operate as lists, then report as strings
        bpdict_SRI[k] = [init]*len(v)
        bpdict_LRI[k] = [init]*len(v)
    return constr_dict, constr_SRI_dict, constr_LRI_dict, bpdict_SRI, bpdict_LRI

def constrUpdate(type, constr_list, vSRI, vLRI, bplist, allp):
    if type == "LRI":
        for pos in allp:
            constr_list[pos] = "x"
        for (p1,p2) in bplist:
            vLRI[p1] = "("
            vLRI[p2] = ")"
    elif type =="SRI":
        for (p1,p2) in bplist:
            constr_list[p1] = "("
            constr_list[p2] = ")"
            vSRI[p1] = "("
            vSRI[p2] = ")"
    else:
        print("ERROR, type not SRI nor LRI")
    return constr_list, vSRI, vLRI

def bpUpdate(seg,type,bplist, bpdict_SRI, bpdict_LRI):
    for (p1,p2) in bplist:
        if type == "LRI":
            bpdict_LRI[seg][p1] = p2
            bpdict_LRI[seg][p2] = p1
        elif type == "SRI":
            bpdict_SRI[seg][p1] = p2
            bpdict_SRI[seg][p2] = p1
        else:
            print("jc.jType not SRI or LRI")
    return bpdict_SRI, bpdict_LRI


def constr_dictUpdate(seg, constr_dict, constr_SRI_dict, constr_LRI_dict, vc, vSRI, vLRI):
    constr_dict[seg] = vc
    constr_SRI_dict[seg] = vSRI
    constr_LRI_dict[seg] = vLRI
    return constr_dict, constr_SRI_dict, constr_LRI_dict
    

def isPknot(seg, bplist, bpdict, constr_list):
    e1,s2 = bplist[-1][0], bplist[-1][-1]
    ss = 0
    if [i for i,j in enumerate(bpdict[seg][e1+1:s2]) if j != -1]: #bp present
        if e1 < min([j for i,j in enumerate(bpdict[seg][e1+1:s2]) if j != -1]) and s2 > max([j for i,j in enumerate(bpdict[seg][e1+1:s2]) if j != -1]):
            #print(f"Not pseudo knot: {list_to_str(constr_list[e1:s2+1])}")
            pass
        else:
            print(f"Is pseudo knot: {list_to_str(constr_list[e1:s2+1])}")

            ss += 1
    else: # bp not present
        print(f"Not pseudo knot: {list_to_str(constr_list[e1:s2+1])}")
    return bool(ss)

def knotUpdate(seg, type, bplist, bpdict_SRI, bpdict_LRI, vSRI, vLRI):
    discard, out_type = 0, ""
    if type == "SRI":
        if isPknot(seg, bplist, bpdict_SRI, vSRI): #knot in SRI
            if isPknot(seg, bplist, bpdict_LRI, vLRI): # knot in LRI
                discard += 1
            else: #not knot in LRI
                out_type = "LRI"
        else: # not knot in SRI
            out_type = "SRI"
    elif type == "LRI":
        if isPknot(seg, bplist, bpdict_LRI, vLRI):
            discard += 1
        else:
            out_type = "LRI"
    else:
        print("jc.jType not SRI or LRI")
    return discard, out_type



            





def getConstrOverlap(pattern,constr_list,ai,bi):
    if re.search("&", pattern) == None:
        s = len(pattern)
        print(f" there is overlap \n pattern   : {pattern} \n constraint: {list_to_str(constr_list[ai:ai+s])}")
    else:
        sa, sb = len(pattern.split("&")[0]), len(pattern.split("&")[1])
        if bi == -1: bi = ai+sa+1
        print(f" there is overlap \n patternA   : {pattern[:sa]} \n constraintA: {list_to_str(constr_list[ai:ai+sa])} \n patternB   : {pattern[sa+1:]} \n constraintB: {list_to_str(constr_list[bi:bi+sb])}")



def getBasePairs(ai, bi, pattern):
    ## get tuples of base pairs       
    s, stack, pairings = len(pattern.split("&")[0]), list(), list()
    if bi == -1: bi = ai+s+1
    for j,p in enumerate(pattern):
        if p == "(": stack.append(j)
        if p == ")": pairings.append((ai+stack.pop(),bi+j-s-1))
    return set(pairings)

def list_to_str(list):
    str = "".join(f"{t}" for t in list)
    return str

def getTestStr(bplist,constr_list):
    setp1, setp2 = zip(*bplist)
    allp = setp1 + setp2 #print(f"setp1: {setp1} \n setp1:{setp2} \n allp:{allp}")
    test = [constr_list[pos] for pos in allp]
    test_str = "".join(f"{t}" for t in test)
    return test_str, allp



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
    return opt

def loadData(fname): 
    ## load data with pickle
    with open(f"{fname}.pcl", "r+b") as pcl_in:
        pcl_data = pickle.load(pcl_in)
    return pcl_data

def saveData(pcl_data, fname):
    ## save data with pickle
    with open(f"{fname}.pcl", "w+b") as pcl_out:
        pickle.dump(pcl_data, pcl_out , protocol=4)


def doCofold(RNA, constraint, opt):
    ## do Cofold
    cvar.dangles = opt.vrd
    cvar.noLP = int(opt.vrn)
    cvar.temperature = opt.vrt
    fc = fold_compound(RNA)
    fc.constraints_add(constraint, CONSTRAINT_DB | CONSTRAINT_DB_DEFAULT)
    pattern, mfe = fc.mfe()
    return mfe, pattern











































































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
