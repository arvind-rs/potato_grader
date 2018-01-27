#!/usr/bin/python

#This is an implementation of a Machine Vision system for grading of potatoes based on quality.
#The system uses Computer Vision and Machine Learning to classify the potatoes into good and rotten sets.
#Author: Arvind Rekha Sura
#Date: 06/12/2016

#Estimated runtime on accompanying image dataset: 16 ~ 18 minutes

#Libraries required: (Suggested to use Anaconda)
#1. cv2
#2. Numpy
#3. Scipy
#4. Scikit-learn

#Steps to install latest scikit library:
# conda create -n sklearn_18 python=2.7
# source activate sklearn_18
# conda install scikit-learn


import numpy as np, scipy.misc, scipy.signal
import time, os, math, csv
from scipy import ndimage


#Global constants
COLOR = "COLOR"
GREYSCALE = "GREYSCALE"
SIZE = (200,125)


def load_image(filename):
	#Function to load the image
	I = scipy.misc.imread(filename)

	return I


def resize_image(image,new_size):
	#Function to resize the image
	import cv2

	I = cv2.resize(image, new_size)

	return I


def segment_image(image):
	#Function to perform image segmentation
	#I have implemented my own image segmentation logic based on region growing segmentation algorithm.
	#This gives me a relatively better segmentation compared to library implemented methods like watershed algorithm for my particular use case.

	import cv2 

	height = image.shape[0]
	width = image.shape[1]

	output = image.copy()

	threshold = 70 #Specifies the color range the candidate pixel can fall under to be differentiated as background.
	
	#Basically I want to segment the potato from the background. For this I've implemented a region growing based segmentation algorithm.
	#I initialize my seed pixel value with the pixel value of the background, whose color range will be very different from the color range of the pixels of the potato (Basically brown potato on white background).
	#To ensure I only take the seed value of the white background, I take seed points at four different locations on the image and take the highest color value amongst the four.
	seed_point_1 = image[height/4,width/4].copy()
	seed_point_2 = image[height*3/4,width/4].copy()
	seed_point_3 = image[height/4,width*3/4].copy()
	seed_point_4 = image[height*3/4,width*3/4].copy()
	
	seed_value = np.amax((seed_point_1,seed_point_2,seed_point_3,seed_point_4),axis=0)
	
	#I maintain a moving average of the color values of the seed value to cover the variations in the background color. 
	moving_average = seed_value.copy()
	values_red = np.array([],dtype='int8')
	values_green = np.array([],dtype='int8')
	values_blue = np.array([],dtype='int8')
	
	for i in range(height):
		for j in range(width):
			
			candidate_pixel = image[i][j].copy()

			if ((candidate_pixel[0] < seed_value[0] + threshold) and (candidate_pixel[0] > seed_value[0] - threshold)) and ((candidate_pixel[1] < seed_value[1] + threshold) and (candidate_pixel[1] > seed_value[1] - threshold)) and ((candidate_pixel[2] < seed_value[2] + threshold) and (candidate_pixel[2] > seed_value[2] - threshold)):
				#If candidate pixel color values fall within specified range then take it as background
				output[i][j] = np.array([0,255,0])

				values_red = np.concatenate([values_red,np.array([candidate_pixel[0]])])
				values_green = np.concatenate([values_green,np.array([candidate_pixel[1]])])
				values_blue = np.concatenate([values_blue,np.array([candidate_pixel[2]])])

				moving_average = np.array([values_red.sum() / values_red.shape[0],values_green.sum() / values_green.shape[0],values_blue.sum() / values_blue.shape[0]])

				seed_value = moving_average.copy()
			else:
				#Else take it as potato
				output[i][j] = np.array([255,0,0])
			
	return output


def pixel_brushing(segmented_image):
	#Function to fill small wrongly segmented gaps in a segmented image.
	#I have implemented this custom algorithm to get better segmentation results with my implementation of region based segmentation algorithm.
	#The algorithm is as follows; Use a sliding window to get the neighbours around a pixel and compute the mode of their color values. The most
	#frequently occuring color value is assigned as the new color value of that pixel. As the sliding window travels across the image, small gaps 
	#in the segmented image get filled to give better segmentation results.
	from scipy import stats

	height = segmented_image.shape[0]
	width = segmented_image.shape[1]

	#Determines the size of the sliding window. It is observed experimently that small values of 'd' give better segmentation results but at the 
	#same time the value should be big enough to cover large gaps. 
	d = 10

	for i in range(d,height-d):
		for j in range(d,width-d):
			current_pixel = segmented_image[i,j]
			neighbours = segmented_image[i-d:i+d+1,j-d:j+d+1]
			mode = stats.mode(neighbours)
			new_pixel_value = stats.mode(mode[0][0])
			segmented_image[i,j] = new_pixel_value[0][0]

	return segmented_image


def multiply(segmented_image,original_image,mode):
	#Function to multiply the segmented image with the original image to extract the foreground potato for further processing
	x_axis = original_image.shape[0]
	y_axis = original_image.shape[1]

	output = original_image.copy()

	for i in range(x_axis):
		for j in range(y_axis):
			current_pixel = segmented_image[i,j].copy()
			if current_pixel[0] == 0 and current_pixel[1] == 255 and current_pixel[2] == 0:
				if mode == "COLOR":
					output[i,j] = np.array([0,0,0])
				elif mode == "GREYSCALE":
					 output[i,j] = 0
				
	return output


def get_defect_pixel_percentage(image):
	#Calculate the ratio of defect to overall pixel values
	#Basic premise is that if the skin color at a pixel on the potato is black, it represents a diseased region.
	#We find the overall diseased region covering the potato and take that as a feature.
	#Similarly, if the white portions inside a potato are visible outside, then the potato is seriously unfit for consumption.
	#We calculate the ratio of white regions as well.   

	import cv2

	output = image.copy()

	x_axis = image.shape[0]
	y_axis = image.shape[1]

	dark_brown = [67,49,35] #diseased brown
	dark_brown_3 = [17,13,10] #black brown
	dark_brown_4 = [102,83,77] #wet brown
	dark_brown_2 = [84,54,30] #light brown
	whiteish_brown = [217,191,141] # whiteish brown / insides of a potato
	whiteish_brown_1 = [242,232,223] # whiteish brown / insides of a potato

	color_leeway = 15 #Specify upper and lower limits of the range the defect pixel color can be
	color_leeway_white = 30 #Specify upper and lower limits of the range the defect pixel color can be

	defect_pixel_count = 0
	overall_pixel_count = 0

	defect_seriousness_0 = 0 #Assign weightage to computed values based on the severity of the defect pixel color.
	defect_seriousness_1 = 1 #Eg. White and dark brown regions are stronger indicators of spoilt potatoes than light brown.
	defect_seriousness_2 = 2 #If you can see the inside white portions of a potato then it's very bad.
	defect_seriousness_3 = 3 

	white_defect_count = 0

	for i in range(x_axis):
		for j in range(y_axis):
			current_pixel = image[i,j]
			if not np.array_equal(current_pixel,np.array([0,0,0])):
				overall_pixel_count += 1 #Maintain a count of overall pixels
				if (current_pixel[0] < (dark_brown[0] + color_leeway) and current_pixel[0] > (dark_brown[0] - color_leeway)) and (current_pixel[1] < (dark_brown[1] + color_leeway) and current_pixel[1] > (dark_brown[1] - color_leeway)) and (current_pixel[2] < (dark_brown[2] + color_leeway) and current_pixel[2] > (dark_brown[2] - color_leeway)):
					output[i,j] = np.array([0,255,0]) #Assigning different colors to output image based on severity of defect pixel for easy debugging. Output image is irrelevant to this function.
					defect_pixel_count += (1 * defect_seriousness_1) #
				if (current_pixel[0] < (dark_brown_2[0] + color_leeway) and current_pixel[0] > (dark_brown_2[0] - color_leeway)) and (current_pixel[1] < (dark_brown_2[1] + color_leeway) and current_pixel[1] > (dark_brown_2[1] - color_leeway)) and (current_pixel[2] < (dark_brown_2[2] + color_leeway) and current_pixel[2] > (dark_brown_2[2] - color_leeway)):
					output[i,j] = np.array([0,125,0])
					defect_pixel_count += (1 * defect_seriousness_0)
				if (current_pixel[0] < (dark_brown_3[0] + color_leeway) and current_pixel[0] > (dark_brown_3[0] - color_leeway)) and (current_pixel[1] < (dark_brown_3[1] + color_leeway) and current_pixel[1] > (dark_brown_3[1] - color_leeway)) and (current_pixel[2] < (dark_brown_3[2] + color_leeway) and current_pixel[2] > (dark_brown_3[2] - color_leeway)):
					output[i,j] = np.array([255,0,0])
					defect_pixel_count += (1 * defect_seriousness_3)
				if (current_pixel[0] < (dark_brown_4[0] + color_leeway) and current_pixel[0] > (dark_brown_4[0] - color_leeway)) and (current_pixel[1] < (dark_brown_4[1] + color_leeway) and current_pixel[1] > (dark_brown_4[1] - color_leeway)) and (current_pixel[2] < (dark_brown_4[2] + color_leeway) and current_pixel[2] > (dark_brown_4[2] - color_leeway)):
					output[i,j] = np.array([0,75,0])
					defect_pixel_count += (1 * defect_seriousness_2)
				if (current_pixel[0] < (whiteish_brown[0] + color_leeway) and current_pixel[0] > (whiteish_brown[0] - color_leeway)) and (current_pixel[1] < (whiteish_brown[1] + color_leeway) and current_pixel[1] > (whiteish_brown[1] - color_leeway)) and (current_pixel[2] < (whiteish_brown[2] + color_leeway) and current_pixel[2] > (whiteish_brown[2] - color_leeway)):
					output[i,j] = np.array([0,0,255])
					defect_pixel_count += (1 * defect_seriousness_3)
					white_defect_count += 1
				if (current_pixel[0] < (whiteish_brown_1[0] + color_leeway_white) and current_pixel[0] > (whiteish_brown_1[0] - color_leeway_white)) and (current_pixel[1] < (whiteish_brown_1[1] + color_leeway_white) and current_pixel[1] > (whiteish_brown_1[1] - color_leeway_white)) and (current_pixel[2] < (whiteish_brown_1[2] + color_leeway_white) and current_pixel[2] > (whiteish_brown_1[2] - color_leeway_white)):
					output[i,j] = np.array([0,0,255])
					defect_pixel_count += (1 * defect_seriousness_3)
					white_defect_count += 1



	if overall_pixel_count != 0: #Edge case check
		defect_area_ratio = float(defect_pixel_count) / float(overall_pixel_count)
		white_defect_area_ratio = float(white_defect_count) / float(overall_pixel_count)
	else:
		defect_area_ratio = 0
		white_defect_area_ratio = 0

	#Calculate the percentage of black and white regions on the potato surface
	defect_region_percentage = round(defect_area_ratio * 100,2)
	white_region_percentage = round(white_defect_area_ratio * 100,2)
	result = {"diseased_region":defect_region_percentage,"white_region":white_region_percentage}

	return result


def get_indentation_count(image):
	#Function to calculate the degree of indentations, cuts and protrusions in the potato.
	#Basic premise is that a relatively smooth potato with no indentations, cuts or protrusions will have low number of pixels in an edge map of that potato.
	#The algorithm is as follows; Extract the edge map of the potato using canny edge detector and count the pixel values on the edge map. Higher number
	#indicates high degree of indentations, cuts or protrusions and vice versa.
	import cv2

	edges = cv2.Canny(image,200,400)

	edge_pixel_count = 0
	for i in range(edges.shape[0]):
		for j in range(edges.shape[1]):
			if edges[i,j] != 0:
				edge_pixel_count += 1

	return edge_pixel_count


def extract_features(directory_dict):
	#Component tasked with extracting features and creating a list of feature vectors

	feature_vector_list = []

	for key in directory_dict.keys():
		current_set = key
		directory_path = directory_dict[key]

		#Set the class label
		if current_set == "bad":
			class_label = 1
		elif current_set == "good":
			class_label = 0

		i = 1
		for root, sub_folders, files in os.walk(directory_path):
			for file in files:
				if '.jpg' in file:
					print "file no.",i
					i += 1
					file_path = os.path.join(root,file)

					print file_path

					print "loading image..."
					input_image_color = load_image(file_path)

					input_image_color = resize_image(input_image_color,SIZE)

					print "segmenting image..."
					segmented_image = segment_image(input_image_color)
					segmented_image = pixel_brushing(segmented_image)

					print "extracting foreground object..."
					foreground_object_color = multiply(segmented_image,input_image_color,COLOR)

					print "getting features..."
					black_pixel_result = get_defect_pixel_percentage(foreground_object_color)
					indentation_count = get_indentation_count(foreground_object_color)

					feature_vector = [file,black_pixel_result["diseased_region"],black_pixel_result["white_region"],indentation_count,class_label]
					feature_vector_list.append(feature_vector)
	
	return feature_vector_list


def create_dataset(feature_vector_list,output_file):
	#Component to create the dataset from the feature vectors
	fp = open(output_file,'w')
	fp.close()

	with open(output_file,"a") as csvfile:
		csv_pointer = csv.writer(csvfile)
		for feature_vector in feature_vector_list:
			csv_pointer.writerow(feature_vector)


def split_dataset(output_file,training_file):
	#Function to split the dataset into training and class label datasets
	fp = open(output_file,'r')

	temp = open(training_file,'w')
	temp.close()

	training_y_csv = open("training_y.csv",'w')
	training_y_csv.close()

	training_fp = open(training_file,'a')

	training_y_fp = open("training_y.csv",'a')

	csv_reader = csv.reader(fp)
	csv_writer_training = csv.writer(training_fp)
	csv_writer_training_y = csv.writer(training_y_fp)
	for row in csv_reader:
		current_row = row
		feature_vector = current_row[1:]
		class_label = current_row[-1]
		csv_writer_training.writerow(feature_vector)
		csv_writer_training_y.writerow(class_label)
		
	training_y_fp.close()
	training_fp.close()
	fp.close()


def run_neural_net(training_file):
	#Function to run the neural network on the dataset
	import pandas as pd
	from sklearn.cross_validation import train_test_split
	from sklearn.neural_network import MLPClassifier
	from sklearn import metrics
	from sklearn.metrics import classification_report,confusion_matrix

	training_csv = pd.read_csv(training_file)
	training_y_csv = pd.read_csv("training_y.csv")
	
	#Split the dataset into training and test sets
	X_train, X_test, y_train, y_test = train_test_split(training_csv, training_y_csv, test_size= 0.4, random_state=42) 
	model = MLPClassifier(hidden_layer_sizes=(2000), activation='logistic',random_state = 42)
	#Fit the neural network
	model.fit(X_train,y_train)
	#Perform classifications
	y_predicted = model.predict(X_test)

	#Calculate accuracy
	accuracy = metrics.accuracy_score(y_test,y_predicted)
	print "\naccuracy = "+str(round(accuracy*100,2))+"%"

	#Get Confusion matrix
	confusion_mat = confusion_matrix(y_test,y_predicted)

	print "\nconfusion matrix..."
	print confusion_mat

	#Calculate Sensitivity and specificity
	print "\nTP\tFP\tFN\tTN\tSensitivity\tSpecificity"
	for i in range(confusion_mat.shape[0]):
		TP = round(float(confusion_mat[i,i]),2)  
		FP = round(float(confusion_mat[:,i].sum()),2) - TP  
		FN = round(float(confusion_mat[i,:].sum()),2) - TP  
		TN = round(float(confusion_mat.sum().sum()),2) - TP - FP - FN
		print str(TP)+"\t"+str(FP)+"\t"+str(FN)+"\t"+str(TN),
		sensitivity = round(TP / (TP + FN),2)
		specificity = round(TN / (TN + FP),2)
		print "\t"+str(sensitivity)+"\t\t"+str(specificity)+"\t\t"


def evaluate_dataset(output_file,training_file):
	#Component tasked with evaluating the dataset
	split_dataset(output_file,training_file)

	run_neural_net(training_file)

def main(directory_dict,training_file,output_file):
	#Main component to start the Machine Vision system

	#Because of dependency conflicts in my installation of scikit-learn I'm not able to install cv2 in my environment hosting the scikit-learn 0.18.1 library.
	#Therefore, I comment out evaluate_dataset() when running extract_features() and comment out extract_features() and create_dataset() when running evaluate_dataset()
	#Please do the same if your implementation of scikit-learn 0.18.1 library does not include cv2.

	#Component to extract features
	feature_vector_list = extract_features(directory_dict)
	
	#Component to create the dataset
	create_dataset(feature_vector_list,output_file)

	#Component to evaluate the dataset
	evaluate_dataset(output_file,training_file)


if __name__ == "__main__":
	start_time = time.time()
	current_directory = os.getcwd()
	directory_good = current_directory + "/data/good_set"
	directory_bad = current_directory + "/data/bad_set"
	directory_dict = {"good":directory_good,"bad":directory_bad}
	training_file = "training.csv"
	output_file = "dataset.csv"

	main(directory_dict,training_file,output_file)
	end_time = time.time()
	print "Runtime : "+str((end_time - start_time) / 60)+" minutes"
