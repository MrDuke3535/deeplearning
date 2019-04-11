from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as  np
import math

def createDataSet():
    data=load_iris()
    dataSet=data.data
    labels=data.target
    newData=[]
    newLabels=[]
    for i in range(len(labels)):
        if(labels[i]!=2):
            newData.append(dataSet[i])
            newLabels.append(labels[i])
    newData=np.mat(newData)
    return norm(newData),newLabels
def norm(dataSet):
    min=dataSet.min(0)
    max=dataSet.max(0)
    range=max-min
    newDataSet = dataSet-min
    newDataSet =newDataSet/range
    return newDataSet
def Relu(x):
    if(x<=0):
        return 0
    else:
        return x
def Sigmoid(x):
    return 1.0/(1+math.exp(-x))
def Relu_derivative(x):
    if x<=0:
        return 0
    else:
        return 1
def train(dataSet,labels,k,learningRate,testSet,testLabels):
    #参数初始化
    n0=dataSet.shape[1]
    hidenLayerNodeNum=3
    w1=np.random.randn(hidenLayerNodeNum,n0)*0.01
    b1=np.zeros((hidenLayerNodeNum,1))
    w2=np.random.randn(1,hidenLayerNodeNum)*0.01
    b2=0
    for i in range(k):
        z1 = np.dot(w1, dataSet.T) + b1
        ReluFun = np.frompyfunc(Relu, 1, 1)
        a1 = ReluFun(z1)
        z2 = np.dot(w2, a1) + b2
        SigmoidFun = np.frompyfunc(Sigmoid, 1, 1)
        a2 = SigmoidFun(z2)
        dz2 = a2 - labels
        m = len(labels)
        dw2 = dz2 * (a1.T) / m
        db2 = np.sum(dz2, axis=1) / m
        Relu_derivativeFun = np.frompyfunc(Relu_derivative, 1, 1)
        dz1 = np.multiply(np.dot(w2.T, dz2), Relu_derivativeFun(z1))
        dw1 = np.dot(dz1, dataSet) / m
        db1 = np.sum(dz1, axis=1)
        w1 = w1 - dw1 * learningRate
        b1 = b1 - db1 * learningRate
        w2 = w2 - dw2 * learningRate
        b2 = b2 - db2 * learningRate
        testData(w1, b1, w2, b2,i+1,testSet,testLabels)
def classify(x):
    if(x>0.5):
        return 1
    else:
        return 0
def testData(w1,b1,w2,b2,index,dataSet,labels):
    # print("w1="+str(w1))
    # print("b1="+str(b1))
    # print("w2="+str(w2))
    # print("b2="+str(b2))
    z1 = np.dot(w1, dataSet.T) + b1
    ReluFun = np.frompyfunc(Relu, 1, 1)
    a1 = ReluFun(z1)
    z2 = np.dot(w2, a1) + b2
    SigmoidFun = np.frompyfunc(Sigmoid, 1, 1)
    a2 = SigmoidFun(z2)
    Classify_Fun=np.frompyfunc(classify,1,1)
    predict=Classify_Fun(a2)
    predict=predict.tolist()
    right=0
    for i in range(len(predict[0])):
        if(predict[0][i]==labels[i]):
            right+=1
    print("第"+str(index)+"次训练正确率:" + str(right/len(labels)))

dataSet,labels=createDataSet()
x_train,x_test,y_train,y_test=train_test_split(dataSet,labels,test_size=0.3,random_state=33)
train(x_train,y_train,5000,0.01,x_test,y_test)