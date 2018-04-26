import pandas as pd 
import numpy, sys
from sklearn.model_selection import train_test_split

DATASET = sys.argv[1]

def readIn (path):
	return pd.read_csv(filepath_or_buffer = path, sep=', ', header='infer')


# def splitData (dataset):
# 	train_data, train_labels 


