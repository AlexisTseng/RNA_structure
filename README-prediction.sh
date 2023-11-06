#/bin/bash

data=/mnt/c/Users/alexi/BoroIntern2023/data
results=/mnt/c/Users/alexi/BoroIntern2023/results
outRNAstr=$results/RNAstr
all=$results/parseSPLASH/all
#mkdir -p $outRNAstr

## Package Requirements
# conda activate RNAfold
# conda install -c bioconda viennarna=2.5.1
# conda install pandas=2.1.0
# conda install scikit-learn=1.30

## installed display from https://www.intel.com/content/www/us/en/download/19344/intel-graphics-windows-dch-drivers.html
# conda install mayavi
## then restarted powershell
    # sudo apt-get update
    # sudo apt-get upgrade

# conda install -c conda-forge umap-learn=0.5.3
# conda install -c conda-forge seaborn=0.12.2
# conda install -c conda-forge hdbscan=0.8.33

## To check package version, use pip show (package name)

#######################################################
## Main Function

#segment=("S11")
#nNeighbors=("300")
#minDistance=("0.0")
#minClusterSize=("50")
#minSamp=("1")

function main(){
    echo "perform intra structure PCA"
    segment=("S11")
    nNeighbors=("15" "30" "50")
    minDistance=("0.0" "0.1")
    minClusterSize=("40" "50" "60")
    minSamp=("10" "15")

    nNeighbors=("15")
    minDistance=("0.0")
    minClusterSize=("60")
    minSamp=("10")
    intraStrPCA segment nNeighbors minDistance minClusterSize minSamp

}

#######################################################
## Functions

function intraStrPCA(){
    local -n uno=$1
    local -n dos=$2
    local -n tre=$3
    local -n qua=$4
    local -n cin=$5
    for i in "${!uno[@]}"
    do
        for a in "${!dos[@]}"
        do
            for b in "${!tre[@]}"
            do
                for c in "${!qua[@]}"
                do
                    for d in "${!cin[@]}"
                    do
                    python RNA_PCAstr_prediction.py --genomeFasta $data/RF-SA11S3.fa --prefix 4_RF-SA11S3_low_20-NSP2_B_jct1.0_jcg3 --parseSPLASHintra $all/4_RF-SA11S3_low_20-NSP2_B.trimmed_structures_intra_chi_scr0.5_scc0.1_scb1_jcs0.9_jct1.0_jca0.05_jcc0.5_jcg3.tsv --segment ${uno[i]} --sampleSize 2000 --matrixVal binary --overwrite --NumComponents 0.9 --nNeighbors ${dos[a]}  --minDistance ${tre[b]} --minClusterSize ${qua[c]} --minSamp ${cin[d]} --colorSettings "#ffffff,#e60202,#222f5b" --junctionRange 3 --junctionCount 0.95
                    done
                done
            done
        done
    done
}

#######################################################
## Call main function

main

exit










