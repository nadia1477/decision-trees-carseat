import pandas as pd 
import numpy, sys, tqdm, random
from sklearn.metrics import mean_squared_error
pd.set_option('expand_frame_repr', False)

class Utils(object):

	def __init__(self, path, targets):
		self.dataset = pd.read_csv(filepath_or_buffer = path, header = 'infer')
		self.target = targets
		self.columnlist = list(self.dataset)
		self.min_rss = None
		self.min_chunk = None
		self.splitFeatures = []
		# enter hyperparameters
		self.alpha = 0.25

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
	regularize = lambda self, data : self.computeRSS(data) + (self.alpha * len(data.columns.tolist()))


	def performSplit (self, train_data, feature):
		# apply split function near mean
		# returns two sub trees
		return train_data[train_data[feature] > train_data[feature].mean()], train_data[train_data[feature] < train_data[feature].mean()]

	# pick min value out of this. 
	def computeRSS (self, chunk):
		# (true of each entry in chunk - median of chunk) ** 2
		rss = 0.0
		predicted_target = self.computeTarget(chunk)
		for index, row in tqdm.tqdm(chunk.iterrows()):
			rss += numpy.power((row[self.target] - predicted_target), 2)
		return rss 

	def displayContents (self):
		print(self.min_rss, self.min_chunk)


	def recursiveSplit (self, train_data, count):

		if len(train_data) < (0.025 * len(self.dataset)):
			return 0

		self.displayContents()
		# generate columns based on the best rss value
		columns = ["CompPrice",  "Income",  "Advertising",  "Population",  "Price",  "Age", "Education"]
		chunks = list()
		rssValues = list()
		choice = random.randint(0, len(columns) - 1)

		t1, t2 = self.performSplit(train_data, columns[choice])
		rssValues.extend((self.computeRSS(t1), self.computeRSS(t2)))
		chunks.extend((t1, t2))

		# find min rss 
		self.min_rss = numpy.argmin(rssValues)
		self.min_chunk = chunks[self.min_rss]

		return self.recursiveSplit (self.min_chunk, count)
		


def main ():

	# read user input for the dataset
	DATASET = sys.argv[1]

	# Create an object for the Utils class with dataset path and target to predict. 
	util = Utils(DATASET, 'Sales')
	train, test = util.process(0.8)

	util.recursiveSplit(train, 0)
	
	# for testing - no RSS computation - directly use left or right traversal and find median at the end

		
if __name__ == '__main__':
	main()

