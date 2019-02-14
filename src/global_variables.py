# -*- coding: utf-8 -*-
"""
HA-Growing Tree-CNN - Source Code
Project 7.4 - Group05

Barbiero Pietro 
Sopegno Chiara
Tavera Antonio 

Created on Sun Aug  5 15:06:05 2018

"""

############################### GLOBAL VARIABLES ################################

#################################################################################

global n_nodes 				#total number of nodes in the tree
global n_classes 			#total number of classes 
global th_class 			#threshold for the classes
global p_value_anova 			#threshold for the Anova
global classes_name 			#vector that contains the name of the classes; the index correspond to the class_id
global interactive_map 			#map that contains all the predefined answers to the user interaction if we are in the automatic mode, useful in the cluster
global interactive 			#boolean used to check if user want interactive mode or not 
global online 				#boolean to set up the online option, load the net if already exist 
global class_separation 		#boolean used to check if the user want to keep the classes together or not 
global debug 				#boolean used to check if we are in the debug mode or not
global anova 				#boolean used to check if the user want the anova active or not 
global automatic_interaction 		#boolean used to check if the automatic interaction is active or not, useful when running on the cluster
global ttest 				#boolean used to activate ttest (instead of softmax) as activation function
global learning_rate 			#learning rate of the training
global batch_size 			#batch size for the training
global epochs 				#maximum number of trainng epochs
