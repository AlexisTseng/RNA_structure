#/bin/bash

data=/mnt/c/Users/alexi/BoroIntern2023/data
results=/mnt/c/Users/alexi/BoroIntern2023/results
outRNAstr=$results/RNAstr
all=$results/parseSPLASH/all

mkdir -p $outRNAstr

############################
## Main Function

function main(){
    echo "Fold RF-SA11S3 intra"
    prefix=("4_RF-SA11S3_low_20-NSP2_B" 
    #"5_RF-SA11S3_high_0-NSP2_A" 
    #"6_RF-SA11S3_high_0-NSP2_B" 
    #"7_RF-SA11S3_high_20-NSP2_A" 
    #"8_RF-SA11S3_high_20-NSP2_B" 
    #"9_RF-SA11S3_low_0-NSP2_nolig_A"
    )
    genfasta=("RF-SA11S3" 
                #"SA11_WT"
                )
    foldSplash genfasta prefix SPtable
    
    #echo "print genfasta"
    #cat $SA11_WT_fa

    #echo "print SPLASH table"
    #cat $all/4_RF-SA11S3_low_20-NSP2_B.trimmed_structures_intra_chi_scr0.5_scc0.1_scb1_jcs0.9_jct3.0_jca0.05_jcc0.5_jcg5.tsv | head -n 30

    #echo "Fold SA11 intra"
    #prefix=("2_transcribing_DLP_B" "3_non-transcribing_DLP_A"  "4_SA11_20-NSP2_B" "4_non-transcribing_DLP_B" "5_RF-S11_0-NSP2"  "5_transcribing_DLP_NSP2" "6_RF-S11-S5_0-NSP2"  "6_transcribing_DLP_A_prox-lig-control_1" "7_RF-S11-S5_20-NSP2" "7_RF-S11-S5_20-NSP2"  "7_transcribing_DLP_A_prox-lig-control_3" "8_RF-S11-S5-S10_0-NSP2" "8_RF-S11-S5-S10_0-NSP2"   "8_old_sample" "8_old_sample"  "9_SA11_0-NSP2_A_control" "9_SA11_0-NSP2_A_control")
    #foldSA11_intra genfasta prefix SPtable
}

#######################################################
## Functions

function foldSplash(){
    local -n one=$1
    local -n two=$2
    local -n tre=$3
    for i in "${!two[@]}"
    do
        python RNAstr.py --genomeFasta $data/${one[0]}.fa --prefix $outRNAstr/${two[i]} --parseSPLASHintra $all/${two[i]}.trimmed_structures_intra_chi_scr0.5_scc0.1_scb1_jcs0.9_jct3.0_jca0.05_jcc0.5_jcg5.tsv --overwrite
        #foldSPS.py -fsa <fasta_genome> -pfx <out_prefix> -psp <SPLASH-table> [options]
    done
}

function foldSA11_intra(){
    local -n one=$1
    local -n two=$2
    local -n tre=$3
    for i in "${!two[@]}"
    do
        python RNAstr.py $SA11_WT_fa ${two[i]} $all/${two[i]}.trimmed_structures_intra_chi_scr0.5_scc0.1_scb1_jcs0.9_jct3.0_jca0.05_jcc0.5_jcg5.tsv
        #foldSPS.py -fsa <fasta_genome> -pfx <out_prefix> -psp <SPLASH-table> [options]
    done
}


#######################################################
## Call main function

main

exit