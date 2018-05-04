import pandas as pd 
import numpy, sys


class Utils(object):

	def __init__(self, path, targets):
		self.dataset = pd.read_csv(filepath_or_buffer = path, header = 'infer')
		self.target = targets

	def process (self, training_set_size):
		# edge case -- 1
		if training_set_size > 1:
			training_set_size = training_set_size / 100
		# randomly shuffle all rows
		self.dataset = self.dataset.sample(frac = 1)
		# split train : test :: training_set_size : 1 - training_set_size ratio
		train_dataset, test_data = numpy.split(self.dataset, [int(len(self.dataset) * training_set_size)])
		# train_targets, test_targets = train_dataset.pop('Sales'), test_data.pop('Sales')
		return train_dataset, test_data

	computeTarget = lambda self, chunk : chunk[self.target].median()

	def performSplit (self, train_data, feature):
		# apply split function near mean
		# returns two sub trees
		return train_data[train_data[feature] > train_data[feature].mean()], train_data[train_data[feature] < train_data[feature].mean()]

	# pick min value out of this. 
	def computeRSS (self, chunk):
		# (true of each entry in chunk - median of chunk) ** 2
		rss = 0.0
		true_target = self.computeTarget(chunk)
		for index, row in chunk.iterrows():
			rss += numpy.power((row[self.target] - true_target), 2)
		return rss 

	def recursiveSplit (self, train_data):

		# generate columns based on the best rss value
		columns = ["CompPrice",  "Income",  "Advertising",  "Population",  "Price",  "Age"]
		chunks = list()
		rssValues = list()
		for column in columns:
			t1, t2 = self.performSplit(train_data, column)
			rssValues.extend((self.computeRSS(t1), self.computeRSS(t2)))
			chunks.extend((t1, t2))

		# print(min(rssValues), rssValues)
		








def main ():

	# read user input for the dataset
	DATASET = sys.argv[1]

	# Create an object for the Utils class with dataset path and target to predict. 
	util = Utils(DATASET, 'Sales')
	train, test = util.process(0.8)

	# t, u = util.performSplit(train, 'Price')
	# print(util.computeRSS(t), util.computeRSS(u))
	# print(util.computeTarget(t))

	util.recursiveSplit(train)

		
if __name__ == '__main__':
	main()

