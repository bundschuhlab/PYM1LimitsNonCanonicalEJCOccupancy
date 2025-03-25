# PYM1LimitsNonCanonicalEJCOccupancy
Code and data for the publication "PYM1 limits non-canonical Exon Junction Complex occupancy in a gene architecture dependent manner to tune mRNA expression"

The essential programs ar contained in the ".py" files. 
	1. ML_functions.py contains a variety of functions used in the other ".oy" files.
	2. overfitting.py contains code for testing to see where overfitting occurs on a given dataset.
	3. permutation_models.py contains code for creating and training a specified number of models.
	4. permutation_important.py contains code for measuring the importance of each feature.

The foldchange data are contained in the ".csv" files.
The feature data are contained in the ".txt" and ".tsv" files. 
	Note: "mart_export_exon.txt" is used to create "Feature_matrix.txt".

Each file beginning with "script" is an example of a bash script for running on a cluster

The three directories are used for storing trained models, data splits, and loss per epoch during training.

