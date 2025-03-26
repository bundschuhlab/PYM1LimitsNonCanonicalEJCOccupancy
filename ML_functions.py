# Copyright (C) <2025>  <The Ohio State University>       

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
# Python file of functions used in ML Codes
#

#
# Note: Bessie was my dog and she was a very good girl.
# 

# necessary imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
import random
import re
import scipy
import math
from itertools import combinations

matplotlib.use("pdf")

# import session_info 
# session_info.show()
print(np.version.version)



# Function for taking an annotation file from Ensembl and calculating the gene architecture 
# features for each gene so that feature selection can then be performed with input data 
# (ex: fold changes). The code was written so new functions calculating new features 
# can be added without too much restructuring (hopefully). Read function info for
# description on file formats and such. 
#
# place = file location of necessary files (ideally they are all in the same folder)
# ensemble_file = file from ensemble (file layout in function below)
# additional_features_file = file with additional features to be added (layout in function below)
# single exon = 1 (true), if set to '1', single exons are included in the feature matrix.
#               They are otherwise excluded. 
def gene_features_creator(place,ensembl_file,additional_features_file,single_exon):
    # this function takes an input files and reads through it and coalesces all
    # the relevant info for a specific gene. In theory the file would contain the 
    # exon locations for each gene and then this file would make a dictionary of every 
    # gene and then the dictionary item would be a list of all the exon intervals for 
    # the gene. function returns a dict of genes whose item is an empty list and dict of 
    # genes with their exons in a list.

    # Keep in mind this code was written with the assumption that only the MANE transcript
    # was selected for analysis. But, the input file can give ENSTs instead of ENSGs to 
    # perform transcript level analysis.
    #
    # The ensembl file format is below. If additional information from ensembl is 
    # needed to calculate desired additional features, that information can be added
    # after the information below and the gene_info function will need to be edited 
    # along with all other functions. The edits most likely will be tedious, but not
    # incredibly hard in a logical and/or technical sense. 
    # 
    # Ensembl file must be of the following format: 
    # Chromosome/scaffold    exon start    exon end    ENSG/ENST    strand

    def gene_info(input_file):

        Gene_info= {}
        Gene_strand = {}

        file1 = open(input_file,'r')
        #A = "Bessie"
        A = file1.readline()
        while A != "":
            A = file1.readline()
            if A == "":
                break
            A = A.split('\t')
            if A[3] in Gene_info:
                Gene_info[A[3]] += [[float(A[1]),float(A[2])]]
            else:
                Gene_info[A[3]] = [[float(A[1]),float(A[2])]]
            A[4] = re.sub('\n',"",A[4])
            Gene_strand[A[3]] = A[4]
        file1.close()

        # This portion of the code ensures that all the intervals (exons)
        # for a gene are ordered 5' to 3' and the reverse strand genes
        # numbers are all made negative to make calculations correct
        Gene_features = {}
        for gene in Gene_info:
            Gene_features[gene] = []
            A = Gene_info[gene]
            if Gene_strand[gene] == "1":
                A = sorted(A)
            else:
                A = sorted(A,reverse=True)
                for i in range(0,len(A)):
                    A[i] = sorted(A[i],reverse=True)
                for i in range(0,len(A)):
                    for j in range(0,2):
                        A[i][j] = -1*A[i][j]
            Gene_info[gene] = A
        
        return Gene_info, Gene_features

    # Calculates the number of exons in a gene
    def exon_number(Gene_info,Gene_features):
        for gene in Gene_info:
            number = len(Gene_info[gene])
            Gene_features[gene] += [number]
        return Gene_features

    # Calculates the number of introns in a given gene
    def intron_number(Gene_info,Gene_features):
        for gene in Gene_info:
            number = len(Gene_info[gene]) - 1
            Gene_features[gene] += [number]
        return Gene_features

    # Calculates the geometirc mean of the exon lengths
    def geometric_mean_exon_length(Gene_info,Gene_features):
        for gene in Gene_info:
            exons = Gene_info[gene]
            factor = len(exons)
            geo_mean = 1
            for exon in exons:
                val = exon[1]-exon[0]+1
                geo_mean = geo_mean*(val**(1/factor))
            Gene_features[gene] += [geo_mean]
        return Gene_features

    # Calculates the geometric mean of the intron lengths
    def geometric_mean_intron_length(Gene_info,Gene_features):
        for gene in Gene_info:
            exons = Gene_info[gene]
            factor = len(exons) - 1
            geo_mean = 1
            if factor == 0:
                geo_mean = 0
            else:
                for i in range(1,len(exons)):
                    val = exons[i][0] - exons[i-1][1] - 1
                    geo_mean = geo_mean*(val**(1/factor))
            Gene_features[gene] += [geo_mean]
        return Gene_features

    # Calculates the length of the longest exon in the gene
    def length_largest_exon(Gene_info,Gene_features):
        for gene in Gene_info:
            Max = 0
            exons = Gene_info[gene]
            for exon in exons:
                val = exon[1] - exon[0] + 1
                if val > Max:
                    Max = val
                else:
                    Bessie = "Bessie is a Good Girl!"
            Gene_features[gene] += [Max]
        print(Bessie)
        return Gene_features

    # Calculates the length of the longest intron in the gene (genes with no introns get 'NA' value)
    def length_largest_intron(Gene_info,Gene_features):
        for gene in Gene_info:
            Max = 0
            exons = Gene_info[gene]
            if len(exons) > 1:
                for i in range(1,len(exons)):
                    val = exons[i][0] - exons[i-1][1] -1
                    if val > Max:
                        Max = val
                    else:
                        Bessie = "Bessie is a Good Girl!"
            else:
                Max = "NA"
            Gene_features[gene] += [Max]
        print(Bessie)
        return Gene_features

    # Calculates the length of the gene
    def gene_length(Gene_info,Gene_features):
        for gene in Gene_info:
            length = Gene_info[gene][-1][1] - Gene_info[gene][0][0] + 1
            Gene_features[gene] += [length]
        return Gene_features

    # Calculates the total length of the exons added together
    def total_exon_length(Gene_info,Gene_features):
        for gene in Gene_info:
            exons = Gene_info[gene]
            total = 0
            for exon in exons:
                total += exon[1] - exon[0] + 1
            Gene_features[gene] += [total]
        return Gene_features

    # Calculates the total length of the introns added together
    def total_intron_length(Gene_info,Gene_features):
        for gene in Gene_info:
            exons = Gene_info[gene]
            total = 0
            for i in range(1,len(exons)):
                total += exons[i][0] - exons[i-1][1] - 1
            Gene_features[gene] += [total]
        return Gene_features

    # Calculates the length of the shortest exon
    def length_shortest_exon(Gene_info,Gene_features):
        for gene in Gene_info:
            exons = Gene_info[gene]
            Min = 10**(9)
            for exon in exons:
                val = exon[1] - exon[0] + 1
                if val < Min:
                    Min = val
                else:
                    Bessie = "Bessie is a Good Girl!"
            Gene_features[gene] += [Min]
        print(Bessie)
        return Gene_features

    # Calculates the length of the shortest intron (genes with no introns get 'NA' value)
    def length_shortest_intron(Gene_info,Gene_features):
        for gene in Gene_info:
            exons = Gene_info[gene]
            Min = 10**(9)
            for i in range(1,len(exons)):
                val = exons[i][0] - exons[i-1][1] - 1
                if val < Min:
                    Min = val
                else:
                    Bessie = "Bessie is a Good Girl!"
            Gene_features[gene] += [Min]
        print(Bessie)
        return Gene_features

    # Calculates arithmetic mean of exon lengths
    def mean_exon_length(Gene_info,Gene_features):
        for gene in Gene_info:
            exons = Gene_info[gene]
            factor = len(exons)
            total = 0
            for exon in exons:
                val = exon[1]-exon[0] + 1
                total = total+val
            mean_length = total/factor
            Gene_features[gene] += [mean_length]
        return Gene_features

    # Calculates arithmetic mean of intron lengths
    def mean_intron_length(Gene_info,Gene_features):
        for gene in Gene_info:
            exons = Gene_info[gene]
            factor = len(exons) - 1
            total = 0
            if factor == 0:
                geo_mean = 0
            else:
                for i in range(1,len(exons)):
                    val = exons[i][0] - exons[i-1][1] - 1
                    total = total+val
                mean_length = total/factor
            Gene_features[gene] += [mean_length]
        return Gene_features

    # Calculates the length of the first intron (genes with no introns get 'NA' value)
    def length_first_intron(Gene_info,Gene_features):
        for gene in Gene_info:
            exons = Gene_info[gene]
            if len(exons) > 1:
                length = exons[1][0]-exons[0][1] - 1
            else:
                length = "NA"
            Gene_features[gene] += [length]
        return Gene_features


    # the following functions do the same as some of the above described functions
    # but they exclude the last exon in the calculation.
    def length_largest_exon_excluding_last(Gene_info,Gene_features):
        for gene in Gene_info:
            Max = 0
            exons = Gene_info[gene]
            for i in range(0,len(exons)-1):
                val = exons[i][1] - exons[i][0] + 1
                if val > Max:
                    Max = val
                else:
                    Bessie = "Bessie is a Good Girl!"
            Gene_features[gene] += [Max]
        print(Bessie)
        return Gene_features

    def total_exon_length_excluding_last(Gene_info,Gene_features):
        for gene in Gene_info:
            exons = Gene_info[gene]
            total = 0
            for i in range(0,len(exons)-1):
                total += exons[i][1] - exons[i][0] + 1
            Gene_features[gene] += [total]
        return Gene_features

    def geometric_mean_exon_length_excluding_last(Gene_info,Gene_features):
        for gene in Gene_info:
            exons = Gene_info[gene]
            factor = len(exons) - 1
            geo_mean = 1
            for i in range(0,len(exons)-1):
                val = exons[i][1]-exons[i][0] + 1
                geo_mean = geo_mean*(val**(1/factor))
            Gene_features[gene] += [geo_mean]
        return Gene_features

    def length_shortest_exon_excluding_last(Gene_info,Gene_features):
        for gene in Gene_info:
            exons = Gene_info[gene]
            Min = 10**(9)
            for i in range(0,len(exons)-1):
                val = exons[i][1] - exons[i][0] + 1
                if val < Min:
                    Min = val
                else:
                    Bessie = "Bessie is a Good Girl!"
            Gene_features[gene] += [Min]
        print(Bessie)
        return Gene_features

    def mean_exon_length_excluding_last(Gene_info,Gene_features):
        for gene in Gene_info:
            exons = Gene_info[gene]
            factor = len(exons) - 1
            total = 0
            if factor == 0:
                mean_length = 0
            else:
                for i in range(len(exons)-1):
                    val = exons[i][1]-exons[i][0] + 1
                    total = total+val
                mean_length = total/factor
            Gene_features[gene] += [mean_length]
        return Gene_features


    # The following code performs the cumulative action of taking the input
    # annotation file and filling in the feature matrix and returning it in a txt file.

    input_file = place+ensembl_file
    Gene_info,Gene_features = gene_info(input_file)
    Gene_features = gene_length(Gene_info,Gene_features)
    Gene_features = exon_number(Gene_info,Gene_features)
    Gene_features = intron_number(Gene_info,Gene_features)
    Gene_features = geometric_mean_exon_length(Gene_info,Gene_features)
    Gene_features = geometric_mean_intron_length(Gene_info,Gene_features)
    Gene_features = length_largest_exon(Gene_info,Gene_features)
    Gene_features = length_largest_intron(Gene_info,Gene_features)
    Gene_features = total_exon_length(Gene_info,Gene_features)
    Gene_features = total_intron_length(Gene_info,Gene_features)
    Gene_features = length_shortest_exon(Gene_info,Gene_features)
    Gene_features = length_shortest_intron(Gene_info,Gene_features)
    Gene_features = mean_exon_length(Gene_info,Gene_features)
    Gene_features = mean_intron_length(Gene_info,Gene_features)
    Gene_features = length_first_intron(Gene_info,Gene_features)
    # excluding last
    Gene_features = length_largest_exon_excluding_last(Gene_info,Gene_features)
    Gene_features = total_exon_length_excluding_last(Gene_info,Gene_features)
    Gene_features = geometric_mean_exon_length_excluding_last(Gene_info,Gene_features)
    Gene_features = length_shortest_exon_excluding_last(Gene_info,Gene_features)
    Gene_features = mean_exon_length_excluding_last(Gene_info,Gene_features)

    # This is where feature names are placed. If you add additional functions to the 
    # above to create new features, name them here. 
    column_names = []
    column_names += ["Gene Length"]
    column_names += ["Exon Number"]
    column_names += ["Intron Number"]
    column_names += ["Exon Length Geometric Mean"]
    column_names += ["Intron Length Geometric Mean"]
    column_names += ["Length of Largest Exon"]
    column_names += ["Length of Largest Intron"]
    column_names += ["Total Exon Length"]
    column_names += ["Total Intron Length"]
    column_names += ["Length of Shortest Exon"]
    column_names += ["Length of Shortest Intron"]
    column_names += ["Mean Exon Length"]
    column_names += ["Mean Intron Length"]
    column_names += ["Length First Intron"]
    column_names += ["Length of Largest Exon Excluding the Last"]
    column_names += ["Total Exon Length Excluding the Last"]
    column_names += ["Exon Length Geometric Mean Excluding the Last"]
    column_names += ["Length of Shortest Exon Excluding the Last"]
    column_names += ["Mean Exon Length Excluding the Last"]

    # Here I have a section of code that adds the features Manu provided. Manu 
    # was a student in molecular genetics that I worked with. This section of
    # code can be used for any tsv file where the first line is geneID followed
    # by the feature names and the following lines are genes and their feature
    # values.  
    def Manu_features(manu_filename,column_names,Gene_features):
        # Adds the new features names to the current features list
        file1 = open(place+manu_filename,'r')
        A = file1.readline()
        A = A.split('\t')
        A[-1] = re.sub('\n',"",A[-1])
        A = A[2:]
        column_names = column_names + A

        # Creates a dictionary with the key as genes and the value as a list
        # of the feature values for that gene. 
        manu_features = {}
        A = "bessie"
        while A != "":
            A = file1.readline()
            if A == "":
                break
            A = A.split('\t')
            gene = A[0]
            A[-1] = re.sub('\n',"",A[-1])
            A = A[2:]
            manu_features[gene] = A
        file1.close()

        # Adds new features to current dict of gene features
        for gene in Gene_features:
            if gene in manu_features:
                Gene_features[gene] = Gene_features[gene] + manu_features[gene]
        
        return Gene_features, column_names

    Gene_features, column_names = Manu_features(additional_features_file, column_names, Gene_features)

    print("number of features = "+str(len(column_names)))

    # Creates the first line of the feature matrix txt file.
    header = "Genes"
    for i in range(0,len(column_names)):
        header = header+'\t'+column_names[i]
    header = header+'\n'

    # This code writes the gene feature values into a text file
    # for use in downstream ML analysis. has option to exclude
    # single exon genes.
    file1 = open(place+"Feature_matrix.txt",'w')
    file1.write(header)
    for gene in Gene_features:
        line = gene
        vals = Gene_features[gene]
        if single_exon == 0:
            if vals[1] > 1:                         # exclude single exon genes
                for i in range(0,len(vals)):
                    line = line+'\t'+str(vals[i])
                line = line+'\n'
                file1.write(line)
        else:
            for i in range(0,len(vals)):
                line = line+'\t'+str(vals[i])
            line = line+'\n'
            file1.write(line)
    file1.close()



# This function creates the features matrix and target vector that are used for
# training and testing the model. It provides information on how 'NA's appearing
# in the input feature matrix affect the number of genes in the resulting output
# feature matrix.
#
# In the future one might want to use something other than foldchanges for 
# prediction. The code would have to be modified to accept a different file 
# format. One could construct a DESeq option and an 'other' option based on 'if'
# statements and the 'other' section of the code could be edited necessarily. 
#
# DESeq2_file = csv results file from DESeq2 run.
# Feature_matrix = feature matrix created by gene_features_creator function above. (
#                  name of file containing all features)
# cutoff = absolute value foldchange cutoff for which genes to include. Genes whose
#          absolute value of their foldchange is below the cutoff are excluded.
# adjpval = adjusted p value cutoff for genes to include. 
# remove_specifics = list of numbers representing the features to be excluded
#                    from the output feature matrix. For example: remove_specifics =
#                    [1,2,3] would remove the 1st three features.
# show = 1 (true). if show = 1, a plot of data loss/retention is shown. 
#
# returns 'features' as a numpy array, 'targets' as a numoy array, and 'column_names'
# as a list of the feature names. 
def create_features_and_targets(place,DESeq2_file,Feature_matrix,cutoff,adjpval,remove_specifics,show,Print):
    
    # puts foldchnage information into dictionary
    file1 = open(place+DESeq2_file,'r')
    A = file1.readline()
    targets_dict = {}
    while A != "":
        A = file1.readline()
        if A == "":
            break
        A = A.split(',')
        A[6] = re.sub('\n',"",A[6])
        A[0] = re.sub('"','',A[0])
        if A[6] != "NA": # filters for genes we have a foldchange for. 
            if float(A[6]) < adjpval: # filters genes according to adjpval cutoff
                if cutoff == 0: # if there is no foldchnage cutoff
                    targets_dict[A[0]] = float(A[2])
                else:
                    if abs(float(A[2])) >= cutoff: # filters for genes above foldchange cutoff
                        targets_dict[A[0]] = float(A[2])
    file1.close()
    
    # goes through feature file and makes a dictionary of genes with the values being
    # a list of all the feature values for that gene (excludes removed features).
    file1 = open(place+Feature_matrix,'r')
    A = file1.readline()
    A = A.split('\t')
    A[-1] = re.sub('\n',"",A[-1])
    length = len(A)
    column_names = A[1:length] # extracts feature names from 1st line
    features_dict = {}
    while A != "":
        A = file1.readline()
        if A == "":
            break
        A = A.split('\t')
        A[-1] = re.sub('\n',"",A[-1])
        #features_dict[A[0]] = []
        has_NA = 0
        current_features = []
        for i in range(1,length):
            if i in remove_specifics: # does not enter feature value info for removed features
                bessie = "good"
            else:
                if A[i] == "NA":
                    current_features += [A[i]]
                    has_NA = 1
                else:
                    current_features += [float(A[i])]
        # if has_NA == 0: # use to only include genes which have a value for each included feature
        #     features_dict[A[0]] = current_features
        features_dict[A[0]] = current_features
        
    file1.close()

    if Print == 1:
        print(" ")
        print("Number of features available: "+str(len(column_names)))
        print(" ")
        print("All available features")
        print(column_names)
        print("")
        print("Excluded features")

    # creates column_names list with only the included features names
    c_names = []
    for i in range(0,len(column_names)):
        if (i+1) in remove_specifics:
            if Print == 1:
                print(str(i+1)+" "+column_names[i])
        else:
            c_names += [column_names[i]]
    column_names = c_names

    # prints the included features
    if Print == 1:
        print(" ")
        print("Features included:")
        print(column_names)
        print(" ")
        print("Length of Features dict (i.e. how many features are available before "+'\n'+
              "intersection with targets dict)")
        print(len(features_dict))
        print(" ")

    
    # removes genes from features_dict that are not in targets_dict.
    genes_to_remove = []
    for gene in features_dict:
        if gene in targets_dict:
            bessie = "good"
        else:
            genes_to_remove += [gene]
    for gene in genes_to_remove:
        features_dict.pop(gene)
    # removes genes from targets_dict that are not in features_dict.
    genes_to_remove = []
    for gene in targets_dict:
        if gene in features_dict:
            bessie = "good"
        else:
            genes_to_remove += [gene]
    for gene in genes_to_remove:
        targets_dict.pop(gene)
    # This ensures we have the union of genes. Genes in each group are
    # based on the ensembl file used to create the input feature matrix
    # and the annotation file used for the DESeq2 analysis. This is why
    # these gene groups can differ in the first place. 
    
    if Print == 1:
        print("Number of features in analysis: "+str(len(column_names)))
        print("Number of genes available for analysis before removal due to 'NA': "+str(len(targets_dict)))
        print("This is the number of genes in the features dict after intersection with the targets dict.")
        print(" ")
    
    
    # goes through and examines data loss when removing NA's (and the columns)
    num_features = len(column_names)
    num_obs_with_NA = 0
    for gene in features_dict:
        for i in range(0,num_features):
            if features_dict[gene][i] == "NA":
                num_obs_with_NA += 1
                break
    if Print == 1:
        print("percent of observations with an 'NA' value: "+str(round((num_obs_with_NA/len(targets_dict))*100,2))+"%")
        print(" ")

    
    # Goes through and calculates data loss after removing each feature
    percent_without_NA = []
    for i in range(0,len(column_names)):
        num_obs_with_NA = 0
        for gene in features_dict:
            A = features_dict[gene][0:i] + features_dict[gene][i+1:]
            for j in range(0,len(A)):
                if A[j] == "NA":
                    num_obs_with_NA += 1    
                    break    
        if Print == 1:
            print(str(i+1)+" percent of observations with an 'NA' value when "+column_names[i]+" is removed: "+str(round((num_obs_with_NA/len(targets_dict))*100,2))+"%")
        percent_without_NA += [100-round((num_obs_with_NA/len(targets_dict))*100,2)]
    print(" ")

    
    # calculates average number of NAs per gene
    num_NAs = 0
    for gene in features_dict:
        for i in range(0,len(features_dict[gene])):
            if features_dict[gene][i] == "NA":
                num_NAs += 1
    if Print == 1:
        print("Average number of NAs per gene: "+str(round(num_NAs/len(targets_dict),2)))
        print(" ")

    
    if show == 1:
        fig = plt.figure(figsize = (14,8))
        plt.plot(column_names,percent_without_NA)
        plt.xticks(rotation=90)
        #plt.tight_layout()
        plt.ylabel("percent")
        plt.title("percent of data without an NA with this column removed")
        plt.savefig(place+"NA_feature_stats.png")
    

    # removes genes from features_dict and targets_dict that have an 
    # NA value for a feature
    genes_to_remove = []
    for gene in features_dict:
        for i in range(0,len(features_dict[gene])):
            if features_dict[gene][i] == "NA":
                genes_to_remove += [gene]
                break
            else:
                bessie = "good"
    for gene in genes_to_remove:
        features_dict.pop(gene)
        targets_dict.pop(gene)
    
    if Print == 1:
        print("Number of genes removed due to NA: "+str(len(genes_to_remove)))
        print("Number of genes left in analysis: "+str(len(features_dict)))
    
    # creates numpy arrays of the features and targets
    Genes = targets_dict.keys()
    genes = []
    for gene in Genes:
        genes += [gene]
    targets = np.zeros(len(targets_dict))
    features = np.zeros([len(targets_dict),len(column_names)])
    for i in range(0,len(genes)):
        targets[i] = targets_dict[genes[i]]
        for j in range(0,len(column_names)):
            features[i,j] = features_dict[genes[i]][j]
    
    # additional code so that outputed features are all those genes that do not
    # a single NA value. 
    # Genes = features_dict.keys()
    # genes = []
    # for gene in Genes:
    #     genes += [gene]
    # features = np.zeros([len(features_dict),len(column_names)])
    # for i in range(0,len(features_dict)):
    #     for j in range(0,len(column_names)):
    #         features[i,j] = features_dict[genes[i]][j]

    if Print == 1:
        print(" ")
        print("Shape of features and targets and len of column_names list")
        print(features.shape,targets.shape,len(column_names))
        print(" ")

    return features, targets, column_names


# This function creates a split of the available data either into a training and 
# test set (80/20) or a training, validation, and test set (80/10/10). 
#
# features = numpy array of feature values for each gene
# targets = numpy array of target for each gene (foldchange)
# howmany = how many splits you want (2 will give train and test, 3 will give
#           train, val, and test).
# place = where you want to store the test sets if they are associated with a certain
#         saved model. set place to the number 0 if you are just needed a split that 
#         won't be used again. You will need to have a folder titled "splits" already
#         made in this location
# step = number denoting what split set you are on (used for split saving). If you train
#        multiple models using multiple splits (in a loop or at different times) put the 
#        designating number here. 
#
# returns the selected splits.
def create_splits(features,targets,howmany,place,model_name,batch_size,num_epochs,learning_rate,step):

    place = place+"splits/"

    num_features = len(features[0,:])
    total = len(targets)

    # scales data in preparation for machine learning
    data_scaler = StandardScaler()
    data_scaler.fit(features)
    scaled_features = data_scaler.transform(features)

    # creates training and testing splits based on 80/20% split
    if howmany == 2:
        test_num = math.floor(.2*total)
        train_num = total-test_num

        choices = [i for i in range(0,total)]
        test_indices = random.sample(choices,test_num)

        training_features = np.zeros([train_num,num_features])
        testing_features = np.zeros([test_num,num_features])
        training_targets = np.zeros(train_num)
        testing_targets = np.zeros(test_num)

        test_place = 0
        train_place = 0
        for i in range(0,total):
            if i in test_indices:
                for j in range(0,num_features):
                    testing_features[test_place,j] = scaled_features[i,j]
                testing_targets[test_place] = targets[i]
                test_place += 1
            else:
                for j in range(0,num_features):
                    training_features[train_place,j] = scaled_features[i,j]
                training_targets[train_place] = targets[i]
                train_place += 1

        # saving testing features and targets (in this case also used as the validation set)
        if os.path.exists(place+"testing_features_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)): 
            os.remove(place+"testing_features_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step))
            np.save(place+"testing_features_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),testing_features)
        else:
            np.save(place+"testing_features_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),testing_features)
        if os.path.exists(place+"testing_targets_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)): 
            os.remove(place+"testing_targets_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step))
            np.save(place+"testing_targets_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),testing_targets)
        else:
            np.save(place+"testing_targets_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),testing_targets)

        # saving training features and targets
        if os.path.exists(place+"training_features_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)): 
            os.remove(place+"training_features_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step))
            np.save(place+"training_features_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),training_features)
        else:
            np.save(place+"training_features_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),training_features)
        if os.path.exists(place+"training_targets_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)): 
            os.remove(place+"training_targets_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step))
            np.save(place+"training_targets_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),training_targets)
        else:
            np.save(place+"training_targets_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),training_targets)
    
        return training_features, testing_features, training_targets, testing_targets
    
    # creates training, validation, and testing splits based on 80/10/10 split.
    if howmany == 3:
        test_num = math.floor(.1*total)
        val_num = math.floor(.1*total)
        train_num = total-(2*test_num)

        choices = [i for i in range(0,total)]
        choices_test = random.sample(choices,test_num+val_num)
        test_indices = random.sample(choices_test,test_num)
        val_indices = []
        for num in choices_test:
            if num in test_indices:
                bessie = "good"
            else:
                val_indices += [num]

        training_features = np.zeros([train_num,num_features])
        validation_features = np.zeros([val_num,num_features])
        testing_features = np.zeros([test_num,num_features])
        training_targets = np.zeros(train_num)
        validation_targets = np.zeros(val_num)
        testing_targets = np.zeros(test_num)

        test_place = 0
        val_place = 0
        train_place = 0
        for i in range(0,total):
            if i in test_indices:
                for j in range(0,num_features):
                    testing_features[test_place,j] = scaled_features[i,j]
                testing_targets[test_place] = targets[i]
                test_place += 1
            elif i in val_indices:
                for j in range(0,num_features):
                    validation_features[val_place,j] = scaled_features[i,j]
                validation_targets[val_place] = targets[i]
                val_place += 1
            else:
                for j in range(0,num_features):
                    training_features[train_place,j] = scaled_features[i,j]
                training_targets[train_place] = targets[i]
                train_place += 1


        # saving testing features and targets
        if os.path.exists(place+"testing_features_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)): 
            os.remove(place+"testing_features_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step))
            np.save(place+"testing_features_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),testing_features)
        else:
            np.save(place+"testing_features_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),testing_features)
        if os.path.exists(place+"testing_targets_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)): 
            os.remove(place+"testing_targets_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step))
            np.save(place+"testing_targets_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),testing_targets)
        else:
            np.save(place+"testing_targets_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),testing_targets)

        # saving training features and targets
        if os.path.exists(place+"training_features_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)): 
            os.remove(place+"training_features_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step))
            np.save(place+"training_features_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),training_features)
        else:
            np.save(place+"training_features_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),training_features)
        if os.path.exists(place+"training_targets_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)): 
            os.remove(place+"training_targets_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step))
            np.save(place+"training_targets_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),training_targets)
        else:
            np.save(place+"training_targets_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),training_targets)
    
        return training_features, validation_features, training_targets, validation_targets



# This function creates the model given the number of features and 
# prints the network structure if 'build' == 1.
#
# num_features = number of features
# build = 1 (true), builds network and prints structure is set to true.
#
# returns the built model
def create_model(num_features,build,learning_rate,optimizer):
    # Model structure
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_features,activation='leaky_relu'),
        tf.keras.layers.Dense(500,activation='leaky_relu'),
        tf.keras.layers.Dense(400,activation='leaky_relu'),
        tf.keras.layers.Dense(300,activation='leaky_relu'),
        tf.keras.layers.Dense(200,activation='leaky_relu'),
        tf.keras.layers.Dense(200,activation='leaky_relu'),
        tf.keras.layers.Dense(200,activation='leaky_relu'),
        tf.keras.layers.Dense(150,activation='leaky_relu'),
        tf.keras.layers.Dense(100,activation='leaky_relu'),
        tf.keras.layers.Dense(50,activation='leaky_relu'),
        tf.keras.layers.Dense(1,activation='linear')
    ])
    
    # Model compiling. This is where you choose the loss function and 
    # the optimizer.
    if optimizer == "adam":
        model.compile(
            loss = tf.keras.losses.MeanSquaredError(
                reduction='sum_over_batch_size',
                name='mean_squared_error'
            ),
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate
            )
        )
    if optimizer == "sgd":
        model.compile(
            loss = tf.keras.losses.MeanSquaredError(
                reduction='sum_over_batch_size',
                name='mean_squared_error'
            ),
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=.9
            )
        )

    model.build((None,num_features))

    # prints model info (network structure)
    if build == 1:
        print(model.summary())

    return model


# This function trains one model and saves the model weights along with the loss arrays. 
# The best model according to the validation loss during training is saved. This function
# is used for estimating when overfitting occurs. 
# 
# Return = 1 (true), if set to 1 will return a model to be used. 
# build = 1 (true), is set to 1, model structure will be printed according to
#         create_model function above. 
# training_features = the training set input data.
# training_targets = the training set target data.
# validation_features = the validation set data.
# validation_targets = the validation set targets.
# num_epochs = the number of epochs to train for.
# batch_size = the batch size to use when training.
# place = file location where the model weights will be stored. A folder titled 'models' will
#         need to be in place already. A folder titled 'loss_arrays' will also be needed.
# model_name = name of series of models or name of single model
# step = model number if you are making multiple in a loop. should just be a number like 1,2,3
def train_models_overfitting(Return,build,training_features,training_targets,validation_features,validation_targets,num_epochs,batch_size,learning_rate,optimizer,place,model_name,step):

    # building model
    num_features = len(training_features[0,:])
    model = create_model(num_features,build,learning_rate,optimizer)

    training_features_tf = tf.constant(training_features,dtype='float32')
    training_targets_tf = tf.constant(training_targets,dtype='float32')

    start = time.time() 
    # train model for given number of epochs
    history = model.fit(training_features_tf, 
        training_targets_tf, 
        batch_size = batch_size, 
        epochs = num_epochs, 
        validation_data = (validation_features, validation_targets),
        verbose=0)
    print(" ")
    print("Time to train: "+str(round((time.time()-start)/60,5)), flush=True)

    # Saving loss arrays for plotting
    training_loss_array = history.history['loss']
    validation_loss_array = history.history['val_loss']

    if os.path.exists(place+"loss_arrays/training_loss_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)+".npy"):
        os.remove(place+"loss_arrays/training_loss_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)+".npy")
        np.save(place+"loss_arrays/training_loss_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),training_loss_array)
    else:
        np.save(place+"loss_arrays/training_loss_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),training_loss_array)
    if os.path.exists(place+"loss_arrays/validation_loss_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)+".npy"):
        os.remove(place+"loss_arrays/validation_loss_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)+".npy")
        np.save(place+"loss_arrays/validation_loss_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),validation_loss_array)
    else:
        np.save(place+"loss_arrays/validation_loss_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),validation_loss_array)

    if Return == 1:
        return model, history


# This function trains one model and saves the model weights along with the loss arrays. 
# The best model according to the validation loss during training is saved. 
# 
# Return = 1 (true), if set to 1 will return a model to be used. 
# build = 1 (true), is set to 1, model structure will be printed according to
#         create_model function above. 
# training_features = the training set input data.
# training_targets = the training set target data.
# validation_features = the validation set data.
# validation_targets = the validation set targets.
# num_epochs = the number of epochs to train for.
# batch_size = the batch size to use when training.
# save_frew = frequency at which models are saved (model saved every # epochs)
# place = file location where the model weights will be stored. A folder titled 'models' will
#         need to be in place already. A folder titled 'loss_arrays' will also be needed.
# model_name = name of series of models or name of single model
# step = model number if you are making multiple in a loop. should just be a number like 1,2,3
def train_models_permutations(Return,build,training_features,training_targets,validation_features,validation_targets,num_epochs,batch_size,learning_rate,optimizer,save_freq,place,model_name,step):

    num_features = len(training_features[0,:])

    # setting the callback for use in reloading the model later
    # (just the weights)
    n_batches = math.ceil(len(training_targets)/batch_size)
    mc = tf.keras.callbacks.ModelCheckpoint(
        place+"models/"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)+".weights.h5",
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=0,
        save_freq = 'epoch',
        save_weights_only=True
    )

    # building the model
    model = create_model(num_features,build,learning_rate,optimizer)

    training_features_tf = tf.constant(training_features,dtype='float32')
    training_targets_tf = tf.constant(training_targets,dtype='float32')

    start = time.time() 
    # train model for given number of epochs
    history = model.fit(training_features_tf, 
        training_targets_tf, 
        batch_size = batch_size, 
        epochs = num_epochs, 
        validation_data = (validation_features, validation_targets),
        callbacks=[mc], 
        verbose=0)
    print(" ")
    print("Time to train: "+str(round((time.time()-start)/60,5)), flush=True)

    saved_model = create_model(num_features,0,learning_rate,optimizer)
    saved_model.load_weights(place+"models/"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)+".weights.h5")
    validation_features_tf = tf.constant(validation_features,dtype='float32')
    validation_targets_tf = tf.constant(validation_targets,dtype='float32')
    initial_loss = saved_model.evaluate(validation_features, validation_targets, verbose=0)
    print("INITIAL VAL LOSS",initial_loss)

    # Saving loss arrays for future plotting if necessary
    training_loss_array = history.history['loss']
    validation_loss_array = history.history['val_loss']
    if os.path.exists(place+"loss_arrays/training_loss_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)+".npy"):
        os.remove(place+"loss_arrays/training_loss_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)+".npy")
        np.save(place+"loss_arrays/training_loss_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),training_loss_array)
    else:
        np.save(place+"loss_arrays/training_loss_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),training_loss_array)
    if os.path.exists(place+"loss_arrays/validation_loss_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)+".npy"):
        os.remove(place+"loss_arrays/validation_loss_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)+".npy")
        np.save(place+"loss_arrays/validation_loss_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),validation_loss_array)
    else:
        np.save(place+"loss_arrays/validation_loss_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step),validation_loss_array)

    if Return == 1:
        return model, history


# This functions takes the "features" matrix created by the create_features_and_targets function
# and determines whether any feature columns are one-hot encoded with other columns. If it finds
# one-hot encoded columns thus indicating a feature takes up more than one column then it returns
# a list of the indices of those columns to be passed to model_permutation_eval so that those columns
# can be permuted together.
#
# features = feature matrix from create_features_and_targets function
#
# returns "one_hot_indices" which is a list of the columns that represent the one hot encoded feature.
def one_hot(features):
    potentials = []
    for i in range(0,len(features[0,:])):
        unique_vals = np.unique(features[:,i])
        unique_vals = np.sort(unique_vals)
        if len(unique_vals) == 2:
            potentials += [i]

    combos = []
    one_hots = []
    for i in range(2,len(potentials)):
        combos += list(combinations(potentials,i))
    for i in range(0,len(combos)):
        combo = np.zeros([len(features[:,0]),len(combos[i])])
        for j in range(0,len(combos[i])):
            for k in range(0,len(features[:,0])):
                combo[k,j] = features[k,combos[i][j]]
        unique_from_combo = np.unique(combo,axis=0)
        if len(unique_from_combo) == len(combos[i]):
        #    print(combos[i],unique_from_combo)
            one_hots += [combos[i]]

    largest_val = 0
    largest_combo_place = "bessie"
    for i in range(0,len(one_hots)):
        if len(one_hots[i]) > largest_val:
            largest_Val = len(one_hots[i])
            largest_combo_place = i

    one_hots = list(one_hots)
    one_hots.sort()

    return one_hots[largest_combo_place]




# This function calculates the percentage loss increase after permutation for the test set
# for each feature, thus measuring feature importance for prediction. It also calculates 
# the kendall tau correlation coefficient between the predicted foldchnages and the feature 
# values for each feature.
#
# place = file location of all information like for above functions
# model_name = the associated model name(s) for the desired models to be tested
# step = which model (if a series of models was created)
# num_p = the number of permutations to perform
# testing = whether or not to use the testing or training set (1/0)
# one_hots = list of columns which are one-hot encoded and must be permuted together
#
# returns 'losses' which is a list of the percentage loss increase for each feature and
# 'direction' which is a list of the above mentioned kendall tau coefficient for each feature. 
def model_permutation_eval(place,model_name,step,num_p,testing,num_features,batch_size,num_epochs,learning_rate,optimizer,one_hots):

    #saved_model = tf.keras.models.load_model(place+"models/"+model_name+"_"+str(step)+".keras",compile=True) 

    # Choosing whether or not to calculate the values using the validation or training set
    if testing == 1:
        testing_features = np.load(place+"splits/testing_features_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)+".npy")
        testing_targets = np.load(place+"splits/testing_targets_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)+".npy")
    else:
        testing_features = np.load(place+"splits/training_features_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)+".npy")
        testing_targets = np.load(place+"splits/training_targets_"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)+".npy")

    # loading the saved model
    saved_model = create_model(num_features,0,learning_rate,optimizer)
    saved_model.load_weights(place+"models/"+model_name+"_"+str(num_features)+"_"+str(batch_size)+"_"+str(num_epochs)+"_"+str(learning_rate)+"_"+str(step)+".weights.h5")

    start = time.time()
    # This chunk of code permutes each feature num_p times and calculates the averaged loss. These
    #  values are then plotted and compared to the original loss and each other to see which 
    #  features impact the loss the most. 
    losses = []
    testing_features_tf = tf.constant(testing_features, dtype='float32')
    testing_targets_tf = tf.constant(testing_targets, dtype='float32')
    initial_loss = saved_model.evaluate(testing_features_tf, testing_targets_tf, verbose=0)
    print("initial loss (test set) of model "+str(step),initial_loss)
    for i in range(0,num_features):
        if i in one_hots:
            testing_features_p = np.copy(testing_features)
            loss_val = 0
            for j in range(0,num_p):
                for k in range(0,len(one_hots)):
                    np.random.shuffle(testing_features_p[:,one_hots[k]])
                testing_features_p_tf = tf.constant(testing_features_p, dtype='float32')
                loss = saved_model.evaluate(testing_features_p_tf, testing_targets_tf, verbose = 0)
                loss_val += loss
            losses += [(((loss_val/num_p)-initial_loss)/initial_loss)*100]
        else:
            testing_features_p = np.copy(testing_features)
            loss_val = 0
            for j in range(0,num_p):
                np.random.shuffle(testing_features_p[:,i])
                testing_features_p_tf = tf.constant(testing_features_p, dtype='float32')
                loss = saved_model.evaluate(testing_features_p_tf, testing_targets_tf, verbose = 0)
                loss_val += loss
            losses += [(((loss_val/num_p)-initial_loss)/initial_loss)*100]
    print(" ", flush=True)
    print("Time to create one permutated loss for each feature: "+str(round((time.time()-start)/60,5)), flush=True)

    # This chunk obtains the predicted foldchanges for each test gene from the model
    # and plots them versus each feature value to obtain direction and strenght 
    # (correlation coef)
    #direction = np.zeros([num_testing_genes,num_features])
    direction = []
    predictions = saved_model.predict(testing_features_tf,
                                      #batch_size=batch_size,
                                      verbose=0)
    #print(predictions.shape)
    for i in range(0,num_features):
        ktau = scipy.stats.kendalltau(predictions,testing_features[:,i])
        direction += [ktau[0]]
    
    actual_direction = []
    for i in range(0,num_features):
        ktau = scipy.stats.kendalltau(testing_targets,testing_features[:,i])
        actual_direction += [ktau[0]]
        

    return losses, direction, actual_direction


