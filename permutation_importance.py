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
# File for calculating the feature importance
#

#
# Note: Read function descriptions in ML_functions.py
#

import ML_functions # type: ignore
import numpy as np
import time
import statistics as stats

# Directory location of all important files and subdirectories. Provide 
# absolute path or if all necessary files are in the same directory as this'
# program and the necessary subdirectories are in this directory, set 
# place = ""
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

# name for models and their associated splits
model_name = "permutation_model_adam"
#model_name = "permutation_model_permuted_adam"
#model_name = "permutation_model_sgd"

# Info needed to create models
batch_size = 256
num_epochs = 2
learning_rate = .000005
number_of_models = 2
optimizer = "adam"
#optimizer = "sgd"

# Info needed for performing importance evalutation and saving the data. 
# num_p = the number of permutations per feature to perform.
# testing = 1 indicates 
num_p = 20
testing = 1

# Creation of same features used to create the models.
features, targets, column_names = ML_functions.create_features_and_targets(place,DESeq2_file,Feature_matrix,cutoff,adjpval,remove_specifics,show,Print)
num_features = len(column_names)
one_hots = ML_functions.one_hot(features)

# Lists to store the loss increases, the correlations coefficients with the predicted
# foldchanges, and the correlation coefficients with the actual foldchnages. 
loss_increases = []
directions = []
actual_directions = []

# Stepping through each model and obtaining the loss increases for each feature as 
# well as the correlation of each feature with the predicted foldchanges. 
for i in range(0,number_of_models):
    step = i
    losses, direction, actual_direction = ML_functions.model_permutation_eval(place,
                                                                              model_name,
                                                                              step,num_p,
                                                                              testing,
                                                                              num_features,
                                                                              batch_size,
                                                                              num_epochs,
                                                                              learning_rate,
                                                                              optimizer,
                                                                              one_hots) 
    loss_increases += [losses]
    directions += [direction]
    actual_directions += [actual_direction]

# Storing all loss increase and correlation information in arrays for
# calculating statistics. 
start = time.time()
num_features = len(loss_increases[0])
permuted_losses_array = np.zeros([len(loss_increases),num_features])
directions_array = np.zeros([len(directions),num_features])
actual_directions_array = np.zeros([len(actual_directions),num_features])
for i in range(0,len(loss_increases)):
    for j in range(0,num_features):
        permuted_losses_array[i,j] = loss_increases[i][j]
        directions_array[i,j] = directions[i][j]
        actual_directions_array[i,j] = actual_directions[i][j]

print(" ", flush=True)
print("Time to create permuted losses and directions arrays: "+str(round((time.time()-start)/60,5)), flush=True)

# Calculation of statistics
start = time.time()
num_iterations = len(permuted_losses_array)
means_losses = []
stds_losses = []
means_directions = []
means_actual_directions = []
X = []
for i in range(0,num_features):
    means_losses += [sum(permuted_losses_array[:,i])/num_iterations]
    stds_losses += [stats.stdev(permuted_losses_array[:,i])]
    means_directions += [sum(directions_array[:,i])/num_iterations]
    means_actual_directions += [sum(actual_directions_array[:,i])/num_iterations]
    X += [i]
print(" ", flush=True)
print("Time to calculate statistics: "+str(round((time.time()-start)/60,5)), flush=True)
print(" ", flush=True)

# Storing of data in a text file.
if testing == 1:
    file1 = open(place+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_plotting_info_testing.txt",'w')
    for i in range(0,len(means_losses)):
        file1.write(str(means_losses[i])+","+str(stds_losses[i])+","+str(means_directions[i])+","+str(means_actual_directions[i])+'\n')
    file1.close()
else:
    file1 = open(place+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_plotting_info_training.txt",'w')
    for i in range(0,len(means_losses)):
        file1.write(str(means_losses[i])+","+str(stds_losses[i])+","+str(means_directions[i])+","+str(means_actual_directions[i])+'\n')
    file1.close()






