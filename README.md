# PYM1LimitsNonCanonicalEJCOccupancy
Code and data for the publication "PYM1 limits non-canonical Exon Junction Complex occupancy in a gene architecture dependent manner to tune mRNA expression"

# Explanation of Repository Contents
## Programs
Additional description of programs and the functions they contain resides in the comments in each file. 
1. ML_functions.py contains a variety of functions used in the other python files.
2. overfitting.py is for testing to see where overfitting occurs on a given dataset.
3. permutation_models.py is for creating and training a specified number of models.
4. permutation_importance.py is for measuring the importance of each feature.
5. FeatureTable.R is for creating the "FeaturesTableV2.tsv" file. 

## Cluster scripts
1. Each file preceded by "script" is an example file for running the above programs on a cluster.

## Data
1. "mart_export_exon.txt" and "FeaturesTableV2.tsv" contain information for creating
	"Feature_matrix.txt" which contains all gene features. 
2. All ".csv" files contain DESeq2 results which contain the foldchanges used as the 
	targets during training. 

## Subdirectories
1. The three subdirectories are used for storing trained models, data splits, and loss 
	per epoch during training.

