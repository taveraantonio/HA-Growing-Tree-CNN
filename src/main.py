# -*- coding: utf-8 -*-
"""
HA-Growing Tree-CNN - Source Code
Project 7.4 - Group05

Barbiero Pietro 
Sopegno Chiara
Tavera Antonio 

Created on Sat Aug  4 19:56:25 2018

"""


from pathlib import Path
import shutil
import numpy as np
from fun_utils import load_dataset, load_test_set
from tree_cnn import TCNode
import sys, os, glob
import global_variables as gv


############################### STR2BOOL #############################################
# convert the input string to the corresponding boolean 
#
#######################################################################################
def str2bool(v):
  return v.lower() in ("yes", "true", "1", "y")



############################### CHECK ERRORS #############################################
# check if new classes already exist, loads the dataset and generates new labels for the 
# dataset in order to avoid same label for different set of images; check other errors
# 
# receive in input the path of the dataset(path) and the name of the new classes 
# returns X_train, y_train, X_test, y_test
#######################################################################################
def check_errors(path, classes):
	# check if new names are already there
	for name in classes:
		if name in gv.classes_name:
			print('Error: One of the new class name is already used. Please try again')
			return None
	
	# load data
	data = load_dataset(input_path=path)
	
	# check if data has been loaded
	if data == None:
		print('Error: No such file or directory')
	X_train, y_train, X_validation, y_validation = data
	
	# generate new labels for the new classes
	new_classes = len(np.unique(y_train))
	# check if the number of new classes matches the class labels provided as input
	if new_classes != len(classes):
		print('Error: The number of provided class names is different from the class labels in the dataset')
		return None
	
	y_train += gv.n_classes
	y_validation += gv.n_classes
	gv.n_classes += new_classes
	
	return X_train, y_train, X_validation, y_validation



############################### MAIN LOOP #############################################
# each iteration display the menu and perform different action 
# according to the user choice (if the test mode is on it does all the things
# automatically):
# 
# if 1, receives two or more new classes to add to the tree. If it is running 
# for the first time it creates the root node and considers the two classes as children, 
# otherwise add the two classes to the root according to the user choice (class block
# or one class at a time) 
#
# if 2 test the network accuracy by receiving a test dataset in input
#
# if 3, the user can change system parameters 
#
# if 4,stop the running program 
#######################################################################################
test_mode = False

# adjustable settings
input_names = ["AC_H", "AD_HH", "S_T_V"]
input_class_separation = [False, False, True]
gv.class_separation = False
gv.th_class = 0.6
gv.interactive = True
gv.automatic_interaction = True
gv.ttest = False 

# more or less fixed settings
gv.online = True
gv.anova = False
gv.p_value_anova = 0.001
gv.learning_rate = 1e-4
gv.batch_size = 100
gv.epochs = 40

gv.classes_name = []
gv.n_nodes = 0
gv.debug = False
gv.n_classes = 0
root = None

output_path = '../data/DATASET_NPY/'
#map each superclass with its known subclasses
gv.interactive_map = {'H': ['HH', 'AD', 'S', 'T', 'V'],
					'AD': ['S', 'T', 'V']}


for file in glob.glob("Node"+"*"+"model_trained.h5"):
	os.remove(file)
	


if not test_mode:
	#USER MODE
	
	while(1):
		print('### Current TreeCNN ###')
		if root != None:
			root.display()
		print('\n################################\n')
	   
		print('*** Menu ***')
		menu_choice = input('1. Add new data\n' \
						 + '2. Test network\n' \
						 + '3. Change system parameters\n' \
						 + '4. Exit the program\n\n' \
						 + '\tInput: ')
		
		if menu_choice == '1':
			#CHOICE "ADD NEW DATA"
			names = input('\nIf you want to add new information, give me the names of the new classes (blank separation): ')
			classes = names.split(' ')
			folder = input('I need also the name of the folder containing the input data, please: ')
			path = '../data/DATASET_NPY/' + folder + '/'		
			
			# load data and check errors
			#save in i the number of the old classes before it will be update from output 
			i = gv.n_classes
			output = check_errors(path, classes)
			
			if output != None:
				#take the output coming from check error functions 
				X_train, y_train, X_validation, y_validation = output
				
				#add the new classes to the list (number of class, classname)
				#save the new class dataset inside a folder
				for name in classes:
					
					path = Path(output_path + str(i))
					if path.is_dir(): 
						shutil.rmtree(output_path + str(i))
					os.mkdir(output_path + str(i))
					
					gv.classes_name.append(name)
					np.save(output_path + str(i) + '/x_train', X_train[y_train == i] )
					np.save(output_path + str(i) + '/x_validation', X_validation[y_validation == i])
					np.save(output_path + str(i) + '/y_train', y_train[y_train == i] )
					np.save(output_path + str(i) + '/y_validation', y_validation[y_validation == i] )
					i += 1
		
				#if there isn't a root create it 
				if root is None:
					#create the tree
					root = TCNode('Node_0')
					gv.n_nodes += 1
					i=0
					for name in classes: 
						root.train_data.append(i) 
						TCNode('Node_'+ str(gv.n_nodes), parent=root, LT=i, class_id=i, data=[])
						gv.n_nodes += 1
						i += 1
						
					# create the convolutional neural network in the root node
					net = root.node_id + '_model_trained.h5'
					model = root.load_net(net, root.train_data)
					root.set_net(net)
					if gv.online: root.set_model(model)
					else: model = None
						
				else:
					if not gv.anova:
						if gv.class_separation: 
							for name in classes:
								#index contains the name of the new class 
								index = gv.classes_name.index(name)
								root = root.add_single_class(root, index)
													
						else:
							#indexes contains the indexes of the new classes
							indexes= []
							for name in classes:
								indexes.append(gv.classes_name.index(name))
							root = root.add_class_block(root, indexes)
					else:
						#anova
						indexes= []
						for name in classes:
							indexes.append(gv.classes_name.index(name))
						root = root.add_class_block_anova(root, indexes)
				
		elif menu_choice == '2':
			#CHOICE "TEST TREE"
			if root is None: 
				print("Error, the model does not exist yet!")
			else:
				names = input('\nGive me the names of the test classes, please (blank separation): ')
				classes = names.split(' ')
				folder = input('I need the name of the folder containing the test data, please: ')
				path = '../data/DATASET_NPY/' + folder + '/'				
				
				#load data
				data = load_test_set(input_path=path)
				
				#check if data has been loaded
				if data == None:
					print('Error: No such file or directory')
				X_test, y_test = data
				
				#arrange indexes with the one corresponding to the class name inserted
				label_indeces = []			
				for index in range(0, len(classes)):
					label_indeces.append(gv.classes_name.index(classes[index]))
					
				for index in range(0, len(y_test)):
					y_test[index] = label_indeces[y_test[index]]
				
				#test accuracy for the classes one by one 
				for i in label_indeces:
					print("Testing class " + gv.classes_name[i] + " ...\n")	
					accuracy = root.test_tree(X_test[y_test == i], y_test[y_test == i])	
					print("Average accuracy: %.4f \n\n" %(accuracy))
					
				#test accuracy for the whole test set received	
				accuracy = root.test_tree(X_test, y_test)
						
				print("Average accuracy for classes " + names + " : %.4f" %(accuracy))
			
			
		elif menu_choice == '3':
			#CHOICE "SET PARAMETER"
			gv.online = input('Do you want the online option? (yes/no)\n\n\tInput: ')
			gv.online = str2bool(gv.online)
				
			gv.interactive = input('Do you want the interactive option? (yes/no)\n\n\tInput: ')
			gv.interactive = str2bool(gv.interactive)
				
			if gv.interactive:
				gv.automatic_interaction = input('Do you want to insert now the known relationships between classes? (yes/no)\n\n\tInput: ')
				gv.automatic_interaction = str2bool(gv.automatic_interaction)
				
				if gv.automatic_interaction:
					cont = True
					while cont:
						names = input( 'Insert a relationship superclass-subclass (blank separation):\n\n\tInput:  ')
						classes = names.split(' ')
						
						if classes[0] not in gv.interactive_map.keys():
							gv.interactive_map.update({classes[0]:[classes[1]]})
						else:
							items = gv.interactive_map.get(classes[0])
							if classes[1] not in items:
								items.append(classes[1])
								
						cont = input('Do you want to continue inserting relationships? (yes/no)\n\n\tInput: ')
						cont = str2bool(cont)
						
						
			gv.anova = input('Do you want the anova option to automatic decide if keeping classes together? (yes/no)\n\n\tInput: ')
			gv.anova = str2bool(gv.anova)
			
			if not gv.anova:
				gv.class_separation = input('Do you want the class separation option? (yes/no)\n\n\tInput: ')
				gv.class_separation = str2bool(gv.class_separation)
			else:
				gv.p_value_anova = float(input('Set the p-value threshold for ANOVA:\n\n\tInput: '))
			
			gv.ttest = input('Do you want the t-test option as activation function? (yes/no)\n\n\tInput: ')
			gv.ttest = str2bool(gv.ttest)
			
			gv.th_class = float(input('Set the class threshold: \n\n\tInput: '))
			
			
		elif menu_choice == '4':
			#CHOICE "EXIT THE PROGRAM"
			sys.exit()
			
		else:
			print('\nI didn\'t understand what do you want from me...')
		
		print('\n\n################################\n')


else:
	#TEST MODE
	#create tree-cnn
	kk=0
	for name in input_names:
		
		classes = name.split("_")
		path = '../data/DATASET_NPY/' + name + '/'
		
		gv.class_separation = input_class_separation[kk]
		kk +=1
				
		# load data and check errors
		#save in i the number of the old classes before it will be update from output 
		i = gv.n_classes
		output = check_errors(path, classes)
		
		if output != None:
			#take the output coming from check error functions 
			X_train, y_train, X_validation, y_validation = output
			
			#add the new classes to the list (number of class, classname)
			#save the new class dataset inside a folder
			for name in classes:
				
				path = Path(output_path + str(i))
				if path.is_dir(): 
					shutil.rmtree(output_path + str(i))
				os.mkdir(output_path + str(i))
				
				gv.classes_name.append(name)
				np.save(output_path + str(i) + '/x_train', X_train[y_train == i] )
				np.save(output_path + str(i) + '/x_validation', X_validation[y_validation == i])
				np.save(output_path + str(i) + '/y_train', y_train[y_train == i] )
				np.save(output_path + str(i) + '/y_validation', y_validation[y_validation == i] )
				i += 1
	
			#if there isn't a root create it 
			if root is None:
				#create the tree
				root = TCNode('Node_0')
				gv.n_nodes += 1
				i=0
				for name in classes: 
					root.train_data.append(i) 
					TCNode('Node_'+ str(gv.n_nodes), parent=root, LT=i, class_id=i, data=[])
					gv.n_nodes += 1
					i += 1
					
				# create the convolutional neural network in the root node
				net = root.node_id + '_model_trained.h5'
				model = root.load_net(net, root.train_data)
				root.set_net(net)
				if gv.online: root.set_model(model)
				else: model = None
					
			else:
				if not gv.anova:
					if gv.class_separation: 
						for name in classes:
							#index contains the name of the new class 
							index = gv.classes_name.index(name)
							root = root.add_single_class(root, index)
							
							print('### Current TreeCNN ###')
							if root != None:
								root.display()
							print('\n################################\n')
												
					else:
						#indexes contains the indexes of the new classes
						indexes= []
						for name in classes:
							indexes.append(gv.classes_name.index(name))
						root = root.add_class_block(root, indexes)
				else:
					#anova
					indexes= []
					for name in classes:
						indexes.append(gv.classes_name.index(name))
					root = root.add_class_block_anova(root, indexes)
					
		print('### Current TreeCNN ###')
		if root != None:
			root.display()
		print('\n################################\n')
	
	#test tree-cnn over all the test sets
	for name in input_names:
		
		classes = name.split("_")
		path = '../data/DATASET_NPY/' + name + '/'
	
		#load data
		data = load_test_set(input_path=path)
			
		#check if data has been loaded
		if data == None:
			print('Error: No such file or directory')
		X_test, y_test = data
		
		#arrange indexes with the one corresponding to the class name inserted
		label_indeces = []			
		for index in range(0, len(classes)):
			label_indeces.append(gv.classes_name.index(classes[index]))
			
		for index in range(0, len(y_test)):
			y_test[index] = label_indeces[y_test[index]]
		
		#test accuracy for the classes one by one 
		for i in label_indeces:
			print("Testing class " + gv.classes_name[i] + " ...\n")	
			accuracy = root.test_tree(X_test[y_test == i], y_test[y_test == i])	
			print("Average accuracy: %.4f \n\n" %(accuracy))
			
		#test accuracy for the whole test set received	
		accuracy = root.test_tree(X_test, y_test)
				
		print("Average accuracy for classes " + name + " : %.4f" %(accuracy))
	


