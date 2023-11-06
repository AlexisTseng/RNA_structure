# RNA_structure
Extracting the major RNA conformations present in a SPLASH sample

RNAstr.py and README-structure.sh are for generating a series of possible RNA structures from non-overlapping interactions present in a given SPLASH dataset
RNA_PCAstr_925.py and README-prediction.sh extract major RNA conformations present in the SPLASH sampling using the following steps:
1. Bootstrap from SPLASH interactions with sampling probability proportional to readcount. Number of sampling attempts is dependent on the RNA length in a quadratic way.
2. Build one RNA structure from the sampled interactions, by sequentially filling the length of the RNA with interactions. If for an incoming interaction, its target nucleotides positions are already paired, then this interaction will be discarded
3. Repeat step 1 and 2, until 2000 RNA structure has been build.
4. Build a base-pairing matrix for the 2000 structures (structure = row, base-pair = column)
5. Apply PCA and/or UMAP on the matrix.
6. Identify clusters using HDBSCAN.
7. Extract the sum base-pairing matrix (position 1 = row, position 2 = column) in each cluster
8. (future direction) Apply graph theory algorithm to extract structure from the matrix in step 7. Region in RNA = node, interaction = edge, read count = weight of an edge. Structure = the maximum matching with the highest weight
