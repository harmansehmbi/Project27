from sklearn.datasets import load_iris
from sklearn import tree


# 1. Lets Create Data Set
irisData =  load_iris()
print("=========IRIS DATASETS===========")
print(irisData)
print(type(irisData))

print()

# Array of Features
print(irisData.data)

print()

# Array of Targets
print(irisData.target)

print()

# Array of Target Names
print(irisData.target_names)

print()

# 2. Lets Create a Model
model = tree.DecisionTreeClassifier()

# 3. Train the Model | Supervised Learning
model.fit(irisData.data, irisData.target)

# 4. Lets Test our Model
inputData = [5.5, 2.6, 4.4, 1.2] # Versicolor Type of Iris Flower

predictedClass = model.predict([inputData])

print()

print(">> Predicted Flower Type is : ", predictedClass)
print(">> Predicted Flower Type is : ", predictedClass[0])
print(">> Predicted Flower Type is : ", irisData.target_names[predictedClass[0]])

# install graphviz