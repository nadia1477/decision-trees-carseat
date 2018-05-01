import pandas as pd 
import numpy, sys



def split_data (dataset):
	
	# randomly shuffle all rows
	dataset = dataset.sample(frac = 1)
	# split train : test :: 80 : 20 ratio
	train_dataset, test_data = numpy.split(dataset, [int(len(dataset) * 0.8)])
	return train_dataset, test_data

class Utils(object):

	def __init__(self, path):
		self.dataset = pd.read_csv(filepath_or_buffer = path, sep=', ', header='infer')

	def split_data (self, training_set_size):
		# edge case -- 1
		if training_set_size > 1:
			training_set_size = training_set_size / 100

		# randomly shuffle all rows
		self.dataset = self.dataset.sample(frac = 1)

		# split train : test :: training_set_size : 1 - training_set_size ratio
		train_dataset, test_data = numpy.split(self.dataset, [int(len(self.dataset) * training_set_size)])
		return train_dataset, test_data

	# def calculate_gini (self, training_data):


class Node(object):
	# defines setters and getters for a node in the tree. 

	def __init__(self, value):
		self.value = value
		self.split = None

	def set_split(self, split):
		self.split = split

	def get_split (self):
		return self.split


def main ():
	# read user input for the dataset
	DATASET = sys.argv[1]
	util = Utils(DATASET)
	train, test = util.split_data(0.8)
	print(train, test)
		
if __name__ == '__main__':
	main()

