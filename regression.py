import numpy
from pprint import pprint
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# HYPERPARAMETERS

TARGET = 'Sales'
FILE = 'carseats.csv'
COLUMNS = ["CompPrice",  "Income",  "Advertising",  "Population",  "Price",  "Age", "Education"]
MAX_HEIGHT = 5
TRAINING_SIZE = 320
FOLDS = 5



pd.set_option('expand_frame_repr', False)	# DEBUG MODE FOR PANDAS DF.

def readCSV(file):

	# RETURNS A DATAFRAME OF DATASET
	return pd.read_csv(file, header = 'infer')

def computeRSS(data):
	# COMPUTING RESIDUAL SUM OF SQUARES
	rss = 0.0
	y_pred = data[TARGET].median() ## this is the regions predicted sales value
	for index, row in data.iterrows():
		rss += numpy.power((row[TARGET] - y_pred), 2)
	return rss

def findBestSplit(data):
	# pprint(data.shape)
	feature = None
	list_rss = list()
	chunk = {}
	for feature in COLUMNS:
		# if  not feature == 'ShelveLoc' or not feature == 'Urban' or not feature == 'US':
		left = data[data[feature] < data[feature].mean()]
		right = data[data[feature] > data[feature].mean()]
		chunk[feature]= {'feature': feature, 'value':data[feature].mean(), 'left' : left, 'right' : right, 'lleaf' : False, 'rleaf': False}
		rss = computeRSS(left) + computeRSS(right)
		list_rss.append(rss)

	min_rss = numpy.argmin(list_rss)

	feature = COLUMNS[min_rss]

	splits = chunk[feature]
	
	# pprint(splits['left'].shape[0])
	# pprint(splits['right'].shape[0])
	return splits ## returns a left and a right dataset to work with recursively. 


leaf = lambda data : numpy.median(data[TARGET])

def buildTree(root, depth):


	# data here is a dictionary of the format -> {'right': df, 'left': df, 'are_leaves': bool}
	if root['lleaf'] == True or root['rleaf'] == True:
		return 

	if root['left'].empty or root['right'].empty:
		root['lleaf'] = True
		root['rleaf'] = True

		# replacing data with median leaf values - predictions
		root['left'] = leaf(root['left'])
		root['right'] = leaf(root['right'])

	if depth == MAX_HEIGHT:
		root['lleaf'] = True
		root['rleaf'] = True

		# replacing data with median leaf values - predictions
		root['left'] = leaf(root['left'])
		root['right'] = leaf(root['right'])

		return

	if root['left'].shape[0] < (TRAINING_SIZE * 0.10):
		root['lleaf'] = True
		root['left'] = leaf(root['left'])
		return
	else:
		to_recurse = findBestSplit(root['left'])
		buildTree(to_recurse, depth + 1)
		root['lleaf'] = True

	if root['right'].shape[0] < (TRAINING_SIZE * 0.10):
		root['rleaf'] = True
		root['right'] = leaf(root['right'])
		return 
	else:
		to_recurse = findBestSplit(root['right'])
		buildTree(to_recurse, depth + 1)
		root['rleaf'] = True


def predict(tree, example):

	feature = tree['feature']
	value = tree['value']

	if example[feature] > value:
		if not tree['rleaf'] == True:
			predict(tree['right'], example)
		else:
			return tree['right'][TARGET].median()
	else:
		if not tree['lleaf'] == True:
			predict(tree['left'], example)
		else:
			return tree['left'][TARGET].median()


def decisionTree(train_data, test_data):


	root = findBestSplit(train_data)
	buildTree(root, 1)
	predictions = list()
	actual = [row[TARGET] for index, row in test_data.iterrows()]
	for index, row in test_data.iterrows():
		pred = predict(root, row)
		# pprint("predicted {}".format(pred))
		predictions.append(pred)

	# pprint(mean_squared_error(actual, predictions))
	pprint("The R2 score for this tree = {}".format(r2_score(actual, predictions)))
	pprint("The mean squared error for this tree = {}".format(mean_squared_error(actual, predictions)))
	return mean_squared_error(actual, predictions)



def main ():
	data = readCSV(FILE)
	train_dataset, test_data = numpy.split(data, [int(len(data) * 0.8)])
	# mse = decisionTree(train_dataset, test_data)
	mses = []
	# BAGGING 
	# collecting results from various trees. 
	for _ in range(10):
		mses.append(decisionTree(train_dataset, test_data))

	pprint(numpy.mean(mses))

	# RANDOM FOREST

	



if __name__ == '__main__':
	main()