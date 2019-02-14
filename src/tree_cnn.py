# -*- coding: utf-8 -*-
"""
HA-Growing Tree-CNN - Source Code
Project 7.4 - Group05

Barbiero Pietro 
Sopegno Chiara
Tavera Antonio 

Created on Sun Aug  5 15:06:05 2018

"""

import os
from anytree import NodeMixin, RenderTree
import scipy
import numpy as np
from sklearn.metrics import accuracy_score
from fun_utils import create_neural_network, train_model, create_neural_network_small
from keras.models import load_model
from pathlib import Path
import global_variables as gv
import shutil
from keras.models import Model
import random

input_path = '../data/DATASET_NPY/'
output_path = '../data/DATASET_NPY/'


def str2bool(v):
	return v.lower() in ("yes", "true", "1", "y")


class TCNode(NodeMixin):  
	############################### INIT ###################################
	# Initializes a node
	#
	# Receive node(self), the parent, the children, the net, the LT, 
	# the class_id, the train data and the model to be initialized
	###########################################################################
	def __init__(self, node_id, parent=None, children=[], net=None, LT=None, class_id = None, data = [], model=None):
		super().__init__()
		self.node_id = node_id
		self.parent = parent
		self.children = children
		self.net = net
		self.LT = LT
		self.train_data = data
		self.model = model
		self.class_id = class_id
	
	
	
	############################### DISPLAY ###################################
	# Display the tree
	#
	# Receive node(self) of the current position
	###########################################################################
	def display(self):
		for pre, _, node in RenderTree(self):
			if node.class_id != None:
				if node.is_leaf: 
					treestr = u"%s%s, Class: %s " % (pre, node.node_id, gv.classes_name[node.class_id])
					print(treestr.ljust(8), end='')
				elif 'Branch' in gv.classes_name[node.class_id]:
					treestr = u"%s%s, Branch: %s " % (pre, node.node_id, gv.classes_name[node.class_id])
					print(treestr.ljust(8), end='')
				else:
					treestr = u"%s%s, Superclass: %s " % (pre, node.node_id, gv.classes_name[node.class_id])
					print(treestr.ljust(8), end='')
					
			else:
				treestr = u"%s%s" % (pre, node.node_id)
				print(treestr.ljust(8), 'children: [', end='')
				for child in node.children:
					print(child.node_id, '', end='')
				print(']', end='')
			
			print('')
			
			
			
	############################### SET NET ###################################
	# Set the net for the node
	#
	# Receive node(self) of the current position and the net to be setted
	###########################################################################		
	def set_net(self, net):
		self.net = net
		
		
		
	############################### SET MODEL##################################
	# Set the model for the node
	#
	# Receive node(self) of the current position and the model to be setted
	###########################################################################
	def set_model(self, model):
		self.model = model
		
		
		
	############################### SET LT ###################################
	# Set the LT for the node
	#
	# Receive node(self) of the current position and the LT(LT) to be setted
	###########################################################################
	def set_LT(self, LT):
		self.LT = LT
		
		
		
	############################### SET PARENT ##################################
	# Set the parent for the node
	#
	# Receive node(self) of the current position and the parent to be setted
	#############################################################################	
	def set_parent(self, parent):
		self.parent = parent
	
	
	
	############################### SET DATA ###################################
	# Set the train data for the specified node
	#
	# Receives as parameters the node(self) of the current position and the data
	# (indexes) to set in that node
	#############################################################################
	def set_data(self, indexes):
		self.train_data = []
		#convert indexes to a list to make it iterable
		if isinstance(indexes, int): 
			indexes = [indexes]
	
		for i in range(0, len(indexes)):
			self.train_data.append(indexes[i])
			
			
		
	############################### LOAD NET ###################################
	# Crete a net if doesn't exist for the current node with the received 
	# train_data. If the net already exist load it from the directory 
	#
	# Receives as parameters the node(self), the name of the net, the train_data 
	# and the retrain parameter to force to retrain the model 
	#############################################################################
	def load_net(self, net_file_name, train_data, retrain = False):
		
		# if you are gv.online, the model exists and you don't need to retrain it, then you can simply return it
		if gv.online and self.model != None and retrain == False:
			return self.model
		
		else:			
			#if offline, at first, try to load the model if it already exists (and you don't need to retrain it)
			net_file = Path(net_file_name)
			
			#convert train data to list, to make it iterable
			if isinstance(train_data, int): 
				train_data = [train_data]
			
			if gv.online == False and self.model == None and retrain == False and net_file.is_file():
				print('Loading CNN...')
				model = load_model(net_file_name)
				print('Loading completed!')
				
			# if not, create the model and save it!
			else:
				self.set_data(train_data)
				#load data
				X = []
				X_val = []
				for i in train_data:
					if len(X) == 0:
						X = np.load(input_path + str(i) + '/x_train.npy')
						y = np.load(input_path + str(i) + '/y_train.npy')
						X_val = np.load(input_path + str(i) + '/x_validation.npy')
						y_val = np.load(input_path + str(i) + '/y_validation.npy')
					else:
						X = np.concatenate((X, np.load(input_path + str(i) + '/x_train.npy')))
						y = np.concatenate((y, np.load(input_path + str(i) + '/y_train.npy')))
						X_val = np.concatenate((X_val, np.load(input_path + str(i) + '/x_validation.npy')))
						y_val = np.concatenate((y_val, np.load(input_path + str(i) + '/y_validation.npy')))
					
				print('Creating a new CNN (online)...')
				img_l, img_h = X.shape[1], X.shape[2]
				if gv.debug == True or len(train_data)==1:
					net = create_neural_network_small(net_file_name, img_l, img_h, len(np.unique(y)))
				else:
					net = create_neural_network(net_file_name, img_l, img_h, len(np.unique(y)))
				model = train_model(net, X, y, X_val, y_val, net_file_name, lr=gv.learning_rate, epochs=gv.epochs, batch_size=gv.batch_size, verbose=1, validation_split=0.25)
				print('Creation completed!')
				
				# if you are gv.online, set the model in the current node before returning it
				if gv.online:
					self.set_model(model)
				
			return model
			
			
			
	############################### TEST MODEL #################################
	# Tests the model 
	# 
	# Receives as parameters the node(self) of the current position, the dataset
	# and the labels 
	# Returns the accuracy
	#############################################################################		
	def test_model(self, X, y):
		# Make a prediction on the test set
		test_predictions = self.net.predict(X)
		test_predictions = np.round(test_predictions)
		
		# Report the accuracy
		accuracy = accuracy_score(y, test_predictions)
		print("Accuracy: " + str(accuracy))
		
		return accuracy
	
	
	
	############################### TEST TREE ##################################
	# Tests the received dataset on the current pretrained tree (self) 
	# 
	# Receives as parameters the tree(self), the dataset to test
	# and the labels 
	# Returns the average score (how many images classify correct)
	#############################################################################
	def test_tree(self, X_test, y_test):
		
		score = []
		for sample in range(0, len(y_test)):
			path = []
			path_pred = self.class_predict(X_test[sample], path)
			if y_test[sample] in path_pred:
				score.append(1)
			else:
				score.append(0)
				
		avg_score = np.mean(score)
		return avg_score
		


	############################### ClASS PREDICT ###############################
	# Predict where to go down in the tree, until the correct class for the 
	# sample is chosen  
	# 
	# Receives as parameters the tree(self), the sample to predict the belonging
	# class and the current path
	# Returns the path
	#############################################################################
	def class_predict(self, sample, path):
		if self.is_leaf:
			path.append(self.class_id)
			return path
		else:
			next_node_number = self.evaluate_net(sample)
			next_node = self.children[next_node_number]
			path.append(self.class_id)
			return next_node.class_predict(sample, path)	


	
	############################### EVALUATE NET ################################
	# Predict where to go down in the tree given the sample
	#  
	# Receives as parameters the tree(self), the sample to predict the node more
	# similar to it 
	# Returns the index corresponding to the next node 
	#############################################################################
	def evaluate_net(self, sample):
		#make a prediction on the test set
		sample = np.expand_dims(sample, axis=0)
		prediction = self.model.predict(sample)		
		#return int(np.round(prediction[0][0]))
		return np.argmax(prediction)

	
		
	
	##################### FLOW NEW CLASSES BELOW #################################
	# If the position where to flow is a leaf node, it puts the new classes below
	# 
	# Receives as parameters the node(self) of the current position, the index
	# where flow the new classes(i1) and the new classes block(indexes)
	# Returns the next node or None
	#############################################################################	
	def flow_new_classes_below(self, i1, indexes):
		
		if self.children[i1].is_leaf == True:
			# create/load CNN model
			net = self.children[i1].node_id + '_model_trained.h5'
			model = self.children[i1].load_net(net, indexes)
			self.children[i1].set_net(net)
			if gv.online: self.children[i1].set_model(model)
			else: model = None
			
			new_class = 0
			for i in indexes:
				TCNode('Node_' + str(gv.n_nodes), parent=self.children[i1], LT=new_class, class_id = i, data = [])
				gv.n_nodes += 1
				new_class += 1
				
			return None
			
		else:
			return i1
	
	
	
	##################### FLOW NEW CLASS BELOW #################################
	# If self.children[i1] is a leaf flow the new classes below, or if it has 
	# only one children flow the class below by retraining the net or create 
	# a branch node. (depends on user input and if the interactive parameter 
	# has been setted); otherwise return the next node
	#
	# Receives as parameters the node(self) of the current position, the index
	# where flow the new class(i1) and the new class(index)
	# Returns the next node or None
	#############################################################################	
	def flow_new_class_below(self, i1, index):
		
		default = False
		if self.children[i1].is_leaf == True:
			default = True
			choice = False
			if gv.interactive: 
				j = self.children[i1].class_id
				if gv.automatic_interaction:
					if gv.classes_name[j] in gv.interactive_map.keys():
						if gv.classes_name[index] in gv.interactive_map[gv.classes_name[j]]:
							choice = True
				else:					
					choice = input('Do you want to add %s as subclass of %s?\nInput(yes/no): '  %(gv.classes_name[index], gv.classes_name[j]))
					choice = str2bool(choice)
					
			if choice:
				#add class as child  
				node = TCNode('Node_' + str(gv.n_nodes), parent=self.children[i1], LT=0, class_id=index, data=[])
				gv.n_nodes += 1
				net = node.node_id + '_model_trained.h5'
				model = self.children[i1].load_net(net, index)
				self.children[i1].set_net(net)
				if gv.online: self.children[i1].set_model(model)
				else: model = None
				
				return None
			
			
		elif (len(self.children[i1].children) == 1): 
			default = True		
			choice = False
			
			if gv.interactive:
				j = self.children[i1].class_id
				if gv.automatic_interaction:
					if gv.classes_name[j] in gv.interactive_map.keys():
						if gv.classes_name[index] in gv.interactive_map[gv.classes_name[j]]:
							choice = True
				else:					
					choice = input('Do you want to add %s as subclass of %s?\nInput(yes/no): ' %(gv.classes_name[index], gv.classes_name[j]))
					choice = str2bool(choice)
				
			if choice: 
				#add the new class as sublass of the current child node
				#train the net again 
				new_train_data = self.children[i1].train_data 
				new_train_data.append(index)
				#retrain the model
				model = self.children[i1].load_net(self.children[i1].net, new_train_data, True)
				if gv.online: self.children[i1].set_model(model)
				else: model = None
				
				TCNode('Node_' + str(gv.n_nodes), parent=self.children[i1], LT=1, class_id = index, data=[])
				gv.n_nodes += 1
				return None

		else:	
			return i1
		
		
		if default: 
			#default case, combine classes below creating branch node 
			self.combine_classes_below(i1, index)
			return None
			

		
	##################### COMBINE CLASSES BELOW ################################
	# Creates a new branch node and combine the beyond_th_indexes classes and 
	# the new one (index) as child of that new node
	#
	# Receives as parameters the node(self) where making the change,the old 
	# classes (beyond_th_indexes) and the new class(index)
	# likelihood 1 and 2 in the evaluate_new_classes_block function
	# Returns None
	#############################################################################	
	def combine_classes_below(self, beyond_th_indexes, index):
		
		output_path = '..\data\DATASET_NPY\\'
		
		try:
			if np.issubdtype(beyond_th_indexes, np.integer):
				beyond_th_indexes = [beyond_th_indexes]
		except:
			beyond_th_indexes = beyond_th_indexes
		
		old_train = []
		for i in beyond_th_indexes: 
			old_train.append(self.children[i].class_id)
		old_train.append(index)
		
		
		#create directory for the new class 
		dir_name = str(gv.n_classes)
		path = Path(output_path + dir_name)
		if path.is_dir(): 
			shutil.rmtree(output_path + dir_name)
		os.mkdir(output_path + dir_name)
		name = 'Branch_' + dir_name
		gv.classes_name.append(name)
		
		id_new_class = gv.n_classes
		gv.n_classes += 1
		
		#combine the dataset coming from the old classes 
		#load data
		X_train = []
		X_val = []
		for i in old_train:
			if len(X_train) == 0:
				X_train = np.load(input_path + str(i) + '/x_train.npy')
				y_train = np.load(input_path + str(i) + '/y_train.npy')
				X_val = np.load(input_path + str(i) + '/x_validation.npy')
				y_val = np.load(input_path + str(i) + '/y_validation.npy')
			else:
				X_train = np.concatenate((X_train, np.load(input_path + str(i) + '/x_train.npy')))
				y_train = np.concatenate((y_train, np.load(input_path + str(i) + '/y_train.npy')))
				X_val = np.concatenate((X_val, np.load(input_path + str(i) + '/x_validation.npy')))
				y_val = np.concatenate((y_val, np.load(input_path + str(i) + '/y_validation.npy')))
				
		indeces = np.arange(0, len(y_train))
		random.shuffle(indeces)
        X_train = X_train[indeces]
        y_train = y_train[indeces]
		lun=int(np.round(len(y_train)/len(old_train)))
		X_train = X_train[:lun]
		y_train = y_train[:lun]
		
		indeces = np.arange(0, len(y_val))
		random.shuffle(indeces)
        X_val = X_val[indeces]
        y_val = y_val[indeces]
		lun=int(np.round(len(y_val)/len(old_train)))
		X_val = X_val[:lun]
		y_val = y_val[:lun]
		
		y_train = np.ones(len(y_train)) * id_new_class
		y_val = np.ones(len(y_val)) * id_new_class
		#save dataset inside the new folder
		np.save(output_path + dir_name + '/x_train', X_train)
		np.save(output_path + dir_name + '/y_train', y_train)
		np.save(output_path + dir_name + '/x_validation', X_val)
		np.save(output_path + dir_name + '/y_validation', y_val)
		
		#train self again with the new train data  
		new_indexes = []
		for i in range(0, len(self.children)):
			if i not in beyond_th_indexes:
				new_indexes.append(self.children[i].class_id)
		new_indexes.append(id_new_class)
		
		#retrain the model
		model =  self.load_net(self.net, new_indexes, True)
		if gv.online: self.set_model(model)
		else: model = None
		
		#create the new node and append it to the father 
		node = TCNode('Node_' + str(gv.n_nodes), parent = self, class_id=id_new_class, data = [])
		gv.n_nodes += 1
		net =  node.node_id + '_model_trained.h5'
		model = node.load_net(net, old_train)
		node.set_net(net)
		if gv.online: node.set_model(model)
		else: model = None
		
		#move self's children to the new node 
		LT_num = 0
		beyond_th_indexes.reverse()
		for i in beyond_th_indexes:
			self.children[i].LT = LT_num
			self.children[i].set_parent(node)
			LT_num += 1
		#create a new node for the new class
		new_node = TCNode('Node_' + str(gv.n_nodes), class_id = index, LT=LT_num, data = [])
		gv.n_nodes += 1
		new_node.set_parent(node)
		
		#rearrange LT of the father 
		i = 0
		for i in range(0, len(self.children)):
			self.children[i].LT = i
			i += 1
		
		return None
	
	
	
	##################### CREATE COMPETITIVE TREE ##############################
	# Creates a new tree for the indexes received and computes the likelihood 
	# matrix considering the old tree
	#
	# Receives as parameters the old tree(self) and the new classes(indexes)
	# Returns the new tree, the max of the likelihood and the index where append 
	# the old tree if ml1>ml2
	#############################################################################	
	def create_competitive_tree(self, indexes):
		
		if isinstance(indexes, int): 
			indexes = [indexes]
			
		root_new = TCNode('Node_' + str(gv.n_nodes))
		gv.n_nodes += 1
		net = root_new.node_id + '_model_trained.h5'
		model = root_new.load_net(net, indexes)
		root_new.set_net(net)
		if gv.online: root_new.set_model(model)
		else: model = None
		
		new_class = 0
		for i in indexes:
			TCNode('Node_' + str(gv.n_nodes), parent=root_new, LT=new_class, class_id = i, data=[])
			gv.n_nodes += 1
			new_class += 1	
		root_new.display()
		
		X_new = []
		y_new = []
		for i in self.train_data:
			if len(X_new) == 0:
				X_new = np.load(input_path + str(i) + '/x_train.npy')
				y_new = np.load(input_path + str(i) + '/y_train.npy')
			else:
				X_new = np.concatenate((X_new, np.load(input_path + str(i) + '/x_train.npy')))
				y_new = np.concatenate((y_new, np.load(input_path + str(i) + '/y_train.npy')))
		
		# compute the likelihood matrix for the new tree 
		L_KM_2 = root_new.compute_likelihood_matrix(X_new, y_new, root_new.train_data)
		max_likelihood_2 = np.max(L_KM_2)
		i2, j2 = np.unravel_index(L_KM_2.argmax(), L_KM_2.shape)
		
		return root_new, max_likelihood_2, i2
	
	
	
	##################### DOVETAIL NEW CLASSES ##############################
	# Swaps trees. If ml1>ml2 appends the old tree(root) to the new one(root_new)
	#
	# Receives as parameters the node(self) where making the change, the old
	# tree(root), the new tree (root_new), the index of the position where to 
	# the append the old tree in the new one, the max received from the 
	# likelihood 1 and 2 in the evaluate_new_classes_block function
	# Returns the root node
	#############################################################################	
	def dovetail_new_classes(self, root, root_new, i2, ml1, ml2):
		
		old_parent = self.parent
		
		if ml2 > ml1:
			#swap the new tree with the old one 
			if old_parent == None :
				root_new.children[i2].set_net(self.net)
				root_new.children[i2].set_model(self.model)
				root_new.children[i2].set_data(self.train_data)
				
				for child in self.children: 
					child.set_parent(root_new.children[i2])
				return root_new
			
			else: 
				root_new.children[i2].set_net(self.net)
				root_new.children[i2].set_model(self.model)
				root_new.children[i2].set_data(self.train_data)
				
				for child in self.children: 
					child.set_parent(root_new.children[i2])
					
				self.set_net(root_new.net)
				self.set_model(root_new.model)
				self.set_data(root_new.train_data)
				
				for child in root_new.children:
					child.set_parent(self)
					
				return root
				
		else:
			length = len(root_new.children)
			gv.n_nodes -= length + 1
			return root
		
		
		
	##################### ADD CLASSES AS SIBLINGS ##############################
	# Adds the classes received as siblings of the existing ones in the current
	# position of the tree
	#
	# Receives as parameters the node(self) where put the classes (indexex) 
	#############################################################################		
	def add_new_classes_as_siblings(self, indexes):
		
		if isinstance(indexes, int):
			indexes = [indexes]
		
		new_class=len(self.children)
		new_indexes = np.concatenate((self.train_data, indexes))
		#retrain the model
		model = self.load_net(self.net, new_indexes, True)
		if gv.online: self.set_model(model)
		else: model = None
		
		# generate the new part of the tree composed of the new siblings
		for i in indexes: 
			TCNode('Node_' + str(gv.n_nodes), parent=self, LT=new_class, class_id=i, data = [])
			gv.n_nodes += 1
			new_class += 1
			
			
			
	##################### EVALUATE NEW CLASSES BLOCK ############################
	# Evaluate the classes block received as parameter;  
	# computes the max of the likelihood and if is higher than the th flows the 
	# block of classes below, otherwise tries to swap the new tree with the old 
	# one; if this fails, add the new classes as sibling of the existing ones 
	#
	# Receives as parameters the node(self) where start appending the classes,
	# the root node(root) and the new classes bloc(indexex) to insert in the tree
	# Returns next_node and the root node 
	#############################################################################	
	def evaluate_new_classes_block(self, root, indexes):
		
		#get the dataset related to the indexes (each index is a new class)
		X_new = []
		y_new = []
		for i in indexes:
			if len(X_new) == 0:
				X_new = np.load(input_path + str(i) + '/x_train.npy')
				y_new = np.load(input_path + str(i) + '/y_train.npy')
			else:
				X_new = np.concatenate((X_new, np.load(input_path + str(i) + '/x_train.npy')))
				y_new = np.concatenate((y_new, np.load(input_path + str(i) + '/y_train.npy')))
		
		#compute the likelihood matrix 
		L_KM_1 = self.compute_likelihood_matrix(X_new, y_new, self.train_data)
		
		#search the max and the position inside the LKM
		max_likelihood_1 = np.max(L_KM_1)
		i1, j1 = np.unravel_index(L_KM_1.argmax(), L_KM_1.shape)
			
		#if max > threshold add the class block in the sub-tree
		if max_likelihood_1 > gv.th_class:
			# the new classes you added will go in the current sub-tree
			next_node = self.flow_new_classes_below(i1, indexes)
			return next_node, root
				
		else:
			# start competition
			root_new, max_likelihood_2, i2 = self.create_competitive_tree(indexes)
				
			if max_likelihood_2 > gv.th_class:
				# case 1: the current subtree (where self==root) can be a subtree of the new tree (where root==root_new)
				# case 2: the new tree (where root=root_new) is sibling of the parent of the current subtree (where self==root)
				root = self.dovetail_new_classes(root, root_new, i2, max_likelihood_1, max_likelihood_2)
				return None, root
				
			else:
				length = len(root_new.children)
				gv.n_nodes -= length + 1
				# the new classes you added will be inserted as siblings of the current node
				self.add_new_classes_as_siblings(indexes)	
				return None, root
	
	
	
	##################### EVALUATE NEW CLASSES BLOCK ANOVA #########################
	# Evaluate the classes block received as parameter; first computes the anova, 
	# if the class separation parameter is true it considers the classes one by
	# one and calls the method add_single_class on the root
	# if the class separation parameter is false it considers the block of classes; 
	# computes the max of the likelihood and if is higher than the th flows the 
	# block of classes below, otherwise tries to swap the new tree with the old 
	# one; if this fails, add the new classes as sibling of the existing ones 
	#
	# Receives as parameters the node(self) where start appending the classes,
	# the root node(root) and the new classes bloc(indexex) to insert in the tree
	# Returns next_node and the root node 
	###############################################################################				
	def evaluate_new_classes_block_anova(self, root, indexes):
		
		#get the dataset related to the indexes (each index is a new class)
		X_new = []
		y_new = []
		for i in indexes:
			if len(X_new) == 0:
				X_new = np.load(input_path + str(i) + '/x_train.npy')
				y_new = np.load(input_path + str(i) + '/y_train.npy')
			else:
				X_new = np.concatenate((X_new, np.load(input_path + str(i) + '/x_train.npy')))
				y_new = np.concatenate((y_new, np.load(input_path + str(i) + '/y_train.npy')))
		
		#compute the likelihood matrix 
		separation, L_KM_1 = self.compute_anova_and_likelihood_matrix(X_new, y_new, self.train_data)
		
		if separation:
			#from here on the classes are inserted separately
			for i in indexes:
				root = root.add_single_class(root, i)
			return None, root
			
		else: 
			#keep classes together	
			#search the max and the position inside the LKM
			max_likelihood_1 = np.max(L_KM_1)
			i1, j1 = np.unravel_index(L_KM_1.argmax(), L_KM_1.shape)
				
			#if max > threshold add the class block in the sub-tree
			if max_likelihood_1 > gv.th_class:
				# the new classes you added will go in the current sub-tree
				next_node = self.flow_new_classes_below(i1, indexes)
				return next_node, root
					
			else:
				# start competition
				root_new, max_likelihood_2, i2 = self.create_competitive_tree(indexes)
					
				if max_likelihood_2 > gv.th_class:
					# case 1: the current subtree (where self==root) can be a subtree of the new tree (where root==root_new)
					# case 2: the new tree (where root=root_new) is sibling of the parent of the current subtree (where self==root)
					root = self.dovetail_new_classes(root, root_new, i2, max_likelihood_1, max_likelihood_2)
					return None, root
					
				else:
					length = len(root_new.children)
					gv.n_nodes -= length + 1
					# the new classes you added will be inserted as siblings of the current node
					self.add_new_classes_as_siblings(indexes)	
					return None, root
		
		
	
	########################### EVALUATE NEW CLASS ################################
	# Evaluate the class received as parameter; if the max of the likelihood
	# is higher than the threshold and if the number of the existing classes 
	# that pass the threshold is higher than 1 it calls the combine classes below 
	# function, else it flows the new class below. Otherwise adds the new class as 
	# siblings of the existing ones 
	#
	# Receives as parameters the node(self) where start appending the class,
	# the root node(root) and the new class(index) to insert in the tree
	# Returns next_node and the root node 
	###############################################################################				
	def evaluate_new_class(self, root, index):
		
		X_new = []
		y_new = []	
		if len(X_new) == 0:
			X_new = np.load(input_path + str(index) + '/x_train.npy')
			y_new = np.load(input_path + str(index) + '/y_train.npy')
		else:
			X_new = np.concatenate((X_new, np.load(input_path + str(index) + '/x_train.npy')))
			y_new = np.concatenate((y_new, np.load(input_path + str(index) + '/y_train.npy')))
		
		L_KM_1 = self.compute_likelihood_matrix(X_new, y_new, self.train_data)
		max_likelihood_1 = np.max(L_KM_1, axis=0)
		max_row_index = L_KM_1.argmax(axis=0)
			
		if max_likelihood_1 > gv.th_class:
			if np.sum(L_KM_1 > gv.th_class) > 1:
				arg_array = np.argwhere(L_KM_1 > gv.th_class)
				beyond_th_indexes = [] 
				for i in range(0, len(arg_array)):  
					beyond_th_indexes.append(arg_array[i][0])
				next_node = self.combine_classes_below(beyond_th_indexes, index)
				return next_node, root 
				
			else:
				next_node = self.flow_new_class_below(max_row_index[0], index)
				return next_node, root
		
		else:
			self.add_new_classes_as_siblings(index)	
			return None, root 

	
	
	########################### ADD CLASS BLOCK ###################################
	# Add a block of classes(indexex) to the tree(self)
	#
	# Receives as parameters the node(self) where start appending the classes,
	# the root node(root) and the new classes block(indexes) to evaluate
	###############################################################################	
	def add_class_block(self, root, indexes):
		
		next_node_number, root = self.evaluate_new_classes_block(root, indexes)
		if next_node_number == None:
			return root
		next_node = self.children[next_node_number]
		return next_node.add_class_block(root, indexes)
	
	
	
	########################### ADD SINGLE CLASS ##################################
	# Add a single class(index) to the tree(self)
	#
	# Receive as parameters the node(self) where start appending the class,
	# the root node(root) and the new class(index) to insert in the tree
	###############################################################################
	def add_single_class(self, root, index):
		 
		next_node_number, root = self.evaluate_new_class(root, index)
		if next_node_number == None:
			return root
		next_node = self.children[next_node_number]
		return next_node.add_single_class(root, index)
	
	
	
	########################### ADD CLASS BLOCK ANOVA #############################
	# Add a block of classes(indexex) to the tree(self)
	# anova calculate if it is more powerful to keep the class together or not
	#
	# Receive as parameters the node(self) where start appending the classes,
	# the root node(root) and the new classes(indexex) to evaluate
	###############################################################################
	def add_class_block_anova(self, root, indexes):
		
		next_node_number, root = self.evaluate_new_classes_block_anova(root, indexes)
		if next_node_number == None:
			return root
		next_node = self.children[next_node_number]
		return next_node.add_class_block_anova(root, indexes)
	
	
	
	########################### COMPUTE LIKELIHOOD MATRIX #########################
	# Compute the likelihood matrix or the ttest (depend on user input)
	#
	# Receives as parameters the node where compute the likelihood matrix,
	# the images, the labels and the indexes of the new classes
	# Returns the likelihood matrix or the ttest 
	###############################################################################
	def compute_likelihood_matrix(self, X, y, indexes):
		
		old_classes = len(self.children)
		new_classes = len(np.unique(y))
		n_samples = int(0.1 * y.shape[0]) 
		
		#example: k=2 x m=2 x i=10
		O_avg = np.zeros((old_classes, new_classes, n_samples))
				
		model = self.load_net(self.net, indexes)
		modelBS = Model(inputs=model.input, outputs=model.get_layer('before_softmax').output)
		model = None
		
		#for each new class create a set of examples
		u = np.unique(y)
		i = 0
		for c in u:
			
			# add new class c
			X_c = X[y==c]
			
			# select 10% of samples for each class
			X_rand = X_c[:n_samples]
			
			predictions = modelBS.predict(X_rand)
			
			if O_avg[:, i, :].shape != predictions.T.shape:
				print('Error')
			
			O_avg[:, i, :] = predictions.T
			
			i += 1
			
		L_KM = np.zeros((old_classes, new_classes))
		
		if not gv.ttest:
			# rows = root classes; columns = new classes
			O_KM = np.mean(O_avg, axis=2)
			
			#better for numerical stability:
			for i in range(new_classes):
				L_KM[:,i] = np.exp(O_KM[:,i] - scipy.misc.logsumexp(O_KM[:,i]))
		
		else:
			#starting t-test analysis
			X_old = []
			
			O_ttest_new = O_avg
			
			O_ttest_old = np.zeros((old_classes,old_classes, n_samples))
			
			i = 0
			for c in indexes:
				X_old = np.load(input_path + str(i) + '/x_train.npy')
				X_rand = X_old[:n_samples]
				
				predictions = modelBS.predict(X_rand)
				
				if O_ttest_old[:, i, :].shape != predictions.T.shape:
					print('Error')
				
				O_ttest_old[:, i, :] = predictions.T
				
				i += 1
			
			normal_distribution = True
			for k in range(old_classes):
				for i in range(new_classes):
					stat, p = scipy.stats.normaltest(O_ttest_new[k,i])
					if p<0.05: normal_distribution = False
			
			for k in range(old_classes):
				for i in range(new_classes):
					
					#Welch test for normal distributions or Mann-Whitney U Test for not normal ones
					if normal_distribution:				
						stat, p_value = scipy.stats.ttest_ind(O_ttest_new[k,i], O_ttest_old[k,k], equal_var = False)
					else:
						stat, p_value = scipy.stats.mannwhitneyu(O_ttest_new[k,i], O_ttest_old[k,k])
					
					L_KM[k,i] = p_value
				
		modelBS = None
		return L_KM
	
	
	
	######################## COMPUTE LIKELIHOOD MATRIX ANOVA ######################
	# Compute the likelihood matrix and the anova to decide if keeping 
	# the classes toghether as a block or not
	#
	# Receives as parameters the node where compute the likelihood matrix,
	# the images, the labels and the indexes of the new classes
	# Returns the likelihood matrix and the classes separation parameters
	###############################################################################
	def compute_anova_and_likelihood_matrix(self, X, y, indexes):
		old_classes = len(self.children)
		new_classes = len(np.unique(y))
		n_samples = int(0.1 * y.shape[0])
		
		# example: k=2 x m=2 x i=10
		O_avg = np.zeros((old_classes, new_classes, n_samples))
		
		
		model = self.load_net(self.net, indexes)
		modelBS = Model(inputs=model.input, outputs=model.get_layer('before_softmax').output)
		
		# for each new class create a set of examples
		u = np.unique(y)
		i = 0
		for c in u:
			
			# add new class c
			X_c = X[y==c]
			y_c = y[y==c]
			
			# select 10% of samples for each class
			X_rand = X_c[:n_samples]
			y_rand = y_c[:n_samples]
			
			predictions = modelBS.predict(X_rand)
			
			if O_avg[:, i, :].shape != predictions.T.shape:
				print('Error')
			
			O_avg[:, i, :] = predictions.T
			
			i += 1
			
		modelBS = None
		model = None
		
		# rows = root classes; columns = new classes
		O_KM = np.mean(O_avg, axis=2)
		L_KM = np.zeros((old_classes, new_classes))
		for i in range(new_classes):
			L_KM[:,i] = np.exp(O_KM[:,i] - scipy.misc.logsumexp(O_KM[:,i]))
		
		separation = False		

		for k in range(old_classes):
			
			normal_distribution = True
			args_anova = [];
			for i in range(new_classes):
				args_anova.append(O_avg[k,i,:30].tolist())
				stat, p = scipy.stats.normaltest(O_avg[k,i,:])
				if p<0.05:
					normal_distribution = False
				
			args = tuple(args_anova)
					
			#One-Way Anova for normal distributions and Kruskal-Wallis for not normal ones
			if normal_distribution:
				F_value, p_value = scipy.stats.f_oneway(*args)
			else:
				F_value, p_value = scipy.stats.kruskal(*args)
			
			if p_value < gv.p_value_anova:
				#reject the null hypotesis that there's not difference between the new classes
				separation = True 
				
				
		return separation, L_KM	
	
	
