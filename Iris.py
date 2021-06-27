
# Required Python Packages
from sklearn.datasets import load_iris      
import numpy as np
from sklearn import tree					

def main():
	dataset = load_iris()  
	
	print("features of dataset")
	print(dataset.feature_names)
	
	print("target names of datasets")
	print(dataset.target_names)		
	
	print("Iris dataset is:")
	
	#Testing block
	index = [1,51,101]	 # 3 data
	test_target = dataset.target[index]
	test_feature =	dataset.data[index]
	
	#Training block
	#we are training 147 data and 3 data for testing out of 150
	#here we are deleting 3 data out of 150 for training them
	train_target = np.delete(dataset.target,index)         
	train_feature = np.delete(dataset.data,index,axis =0)
	
	obj = tree.DecisionTreeClassifier()
	obj.fit(train_feature,train_target)
	
	result = obj.predict(test_feature)
	
	print("Result prediction by ML", result)
	
	print("Result expected",test_target)
	
	
if __name__== "__main__":
	main()
