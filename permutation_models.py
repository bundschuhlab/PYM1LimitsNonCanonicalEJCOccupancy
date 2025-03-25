# Copyright (C) <2022>  <The Ohio State University>       

# This program is free software: you can redistribute it and/or modify                              
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or    
# (at your option) any later version.                                                                                       
# This program is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of           
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the       
# GNU General Public License for more details.                                                                             

# You should have received a copy of the GNU General Public License 
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


#
# File for creating the trained models
#

#
# Note: Read function descriptions in ML_functions.py
#

import ML_functions # type: ignore

# Creation of feature matrix
# set place to the absolute path or if the necessary files are in the same
# directory, set place = "". If they are in a subdirectory, set place to 
# the subdirectory. Provide the ensemble file and the additional features file
# for creating the models. 
# This section of code need only be run once and can be commented out for 
# future runs using the same feature matrix file. 
place = ""
ensembl_file = "mart_export_exon.txt"
additional_features_file = "FeaturesTableV2.tsv"
ML_functions.gene_features_creator(place,ensembl_file,additional_features_file,0)


# File location of all information and info to be stored. Provide absolute path 
# or ensure that the needed files are in the same directory as this program and
# set place = "". If files are in a subdirectory, set place to the subdirectory.
place = ""

# Info needed to create features and targets. Provide the 
# 1. name of the DESeq2_file to be used
# 2. name of the feature matrix (the name will be Feature_Matrix.txt if the 
#    code above is used to generate it)
# 3. the cutoff value to filter for certain magnitudes of foldchanges/targets
# 4. the cutoff value to filter for adjusted p-values below teh cutoff
# 5. the list of column_indices of which features should be removed from 
#    the particular analysis at hand
# 6. the value of print to show or not show stats on the feature data
#    (Print = 1 will show these).
# 7. the value of show to create a plot of the 'NA' feature stats 
#    show=1 will create the plot. 
DESeq2_file = "all_CHX_res.csv"
Feature_matrix = "Feature_matrix.txt"
cutoff = 0
adjpval = 100
remove_specifics = [3,20,36,37,38,39,40,41,42,43]
Print = 0
show = 0

# Info needed to create splits. 3 = training/validation/testing (80/10/10) and 
# 2 = training/validation (80/20)
howmany = 3

# info needed to create models (return = 1 will return the model)
# (build = 1 will print the model structure/architecture)
Return = 0
num_epochs = 2
batch_size = 256
learning_rate = .000005
save_freq = 1
build = 0
optimizer = "adam"
num_models = 2

# name for models and their associated splits
model_name = "permutation_model_"+optimizer
#model_name = "permutation_model_permuted_"+optimizer

features, targets, column_names = ML_functions.create_features_and_targets(place,DESeq2_file,Feature_matrix,cutoff,adjpval,remove_specifics,show,Print)
num_features = len(column_names)

# Place where i shuffle targets if I am making a permuted model
#np.random.shuffle(targets)

for i in range(0,num_models):
    step = i
    training_features,validation_features,training_targets,validation_targets = ML_functions.create_splits(features,
                                                                                                           targets,
                                                                                                           howmany,
                                                                                                           place,
                                                                                                           model_name,
                                                                                                           batch_size,
                                                                                                           num_epochs,
                                                                                                           learning_rate,
                                                                                                           step)
    ML_functions.train_models_permutations(Return,
                                           build,
                                           training_features,
                                           training_targets,
                                           validation_features,
                                           validation_targets,
                                           num_epochs,
                                           batch_size,
                                           learning_rate,
                                           optimizer,
                                           save_freq,
                                           place,
                                           model_name,
                                           step)












