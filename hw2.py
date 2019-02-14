# course: TCSS555
# Homework 2
# date: 10/16/2018
# name: Martine De Cock
# description: Training and testing decision trees with discrete-values attributes

import sys
import math
import pandas as pd


class DecisionNode:

    # A DecisionNode contains an attribute and a dictionary of children. 
    # The attribute is either the attribute being split on, or the predicted label if the node has no children.
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}

    # Visualizes the tree
    def display(self, level = 0):
        if self.children == {}: # reached leaf level
            print(": ", self.attribute, end="")
        else:
            for value in self.children.keys():
                prefix = "\n" + " " * level * 4
                print(prefix, self.attribute, "=", value, end="")
                self.children[value].display(level + 1)
     
    # Predicts the target label for instance x
    def predicts(self, x):
        if self.children == {}: # reached leaf level
            return self.attribute
        value = x[self.attribute]
        subtree = self.children[value]
        return subtree.predicts(x)


# Illustration of functionality of DecisionNode class
def funTree():
    myLeftTree = DecisionNode('humidity')
    myLeftTree.children['normal'] = DecisionNode('no')
    myLeftTree.children['high'] = DecisionNode('yes')
    myTree = DecisionNode('wind')
    myTree.children['weak'] = myLeftTree
    myTree.children['strong'] = DecisionNode('no')
    return myTree

def getPossibleValuesFeatureHas(examples, attribute):
    return examples[attribute].unique()

def calculateEntropy(examples, target):
    possibleTargetValues = getPossibleValuesFeatureHas(examples, target)
    entropyValue = 0
    totalCount = len(examples)
    for possibleTargetValue in possibleTargetValues:
        examplesWithTargetValue = examples[examples[target] == possibleTargetValue]
        probability = len(examplesWithTargetValue)/totalCount
        entropyValue +=  (-1 * probability * math.log(probability, 2))
#        print("count = {} for  value = {}".format(len(examplesWithTargetValue), possibleTargetValue))
    return entropyValue    

def calculateGain(examples, target, attribute):
    entropyForAttribute = calculateEntropy(examples, target)
    possibleAttributeValues = getPossibleValuesFeatureHas(examples, attribute)
    total = 0
    for possibleAttributeValue in possibleAttributeValues:
        subsetExamples = examples[examples[attribute] == possibleAttributeValue]
        subsetEntropy = calculateEntropy(subsetExamples, target) # the attribute is the samne, we are just looking at a subset.
        numerator = len(subsetExamples)
        total += ((numerator/len(examples) * subsetEntropy))
    return entropyForAttribute - total

def determineBestFeature(examples, target, attributes):
    featureToSelectIndex = 0
    largestGain = -100000
    for index, attribute in enumerate(attributes):
        gain = calculateGain(examples, target, attribute)
#        print("Gain = {} for attribute = {}".format(gain, attribute))
        if gain > largestGain:
            featureToSelectIndex = index
            largestGain = gain
#    print("Attribute with highest gain = {}".format(attributes[featureToSelectIndex]))
    return attributes[featureToSelectIndex]

def getMajorityClass(examples, target):
    possibleTargetValues =  getPossibleValuesFeatureHas(examples, target)
    countOfValues = 0
    classValue =  None
    for possibleTargetValue in possibleTargetValues:
        examplesWithTargetValue = examples[examples[target] == possibleTargetValue]
        if len(examplesWithTargetValue) > countOfValues:
            countOfValues = len(examplesWithTargetValue)
            classValue = possibleTargetValue
    return classValue


def id3(examples, target, attributes):
    countOfExamples = len(examples)
#    print("Checking if all examples belong to one class..")
    possibleTargetValues =  getPossibleValuesFeatureHas(examples, target)
    # Check if all examples are in one class.
    for possibleTargetValue in possibleTargetValues:
        examplesWithTargetValue = examples[examples[target] == possibleTargetValue]
        if(len(examplesWithTargetValue) == countOfExamples):
            return DecisionNode(possibleTargetValue)

    feature = determineBestFeature(examples, target, attributes)
#    print("The best feature is: {}".format(feature))
    rootNode = DecisionNode(feature)
    possibleAttributeValues = getPossibleValuesFeatureHas(examples, feature)
    for possibleValue in possibleAttributeValues:
#        print("Looking at possible value of = {}".format(possibleValue))
        subsetOfExamplesForAttribute = examples[examples[feature] == possibleValue]
        if len(subsetOfExamplesForAttribute) != 0:
#            print("Call id3")
            rootNode.children[possibleValue] = id3(subsetOfExamplesForAttribute, target, attributes)
        else:
#            print("choose majority class")
            majorityClass = getMajorityClass(subsetOfExamplesForAttribute, target)
            rootNode.children[possibleValue] = DecisionNode(majorityClass)
    return rootNode


####################   MAIN PROGRAM ######################

# Reading input data
train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])
target = sys.argv[3]
attributes = train.columns.tolist()
attributes.remove(target)


# Learning and visualizing the tree
tree = id3(train,target,attributes)
tree.display()

# Evaluating the tree on the test data
correct = 0
for i in range(0,len(test)):
    if str(tree.predicts(test.loc[i])) == str(test.loc[i,target]):
        correct += 1
print("\nThe accuracy is: ", correct/len(test))