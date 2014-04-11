# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_ml/py_knn/py_knn_opencv/py_knn_opencv.html

import numpy as np
import cv2
from matplotlib import pyplot as plt

def main3():
    img = cv2.imread('digits.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Now we split the image to 5000 cells, each 20x20 size
    cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

    # Make it into a Numpy array. It size will be (50,100,20,20)
    x = np.array(cells)

    # Now we prepare train_data and test_data.
    train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
    test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

    # Create labels for train and test data
    k = np.arange(10)
    train_labels = np.repeat(k,250)[:,np.newaxis]
    test_labels = train_labels.copy()

    # Initiate kNN, train the data, then test it with test data for k=1
    knn = cv2.KNearest()
    knn.train(train,train_labels)
    ret,result,neighbours,dist = knn.find_nearest(test,k=5)

    # Now we check the accuracy of classification
    # For that, compare the result with test_labels and check which are wrong
    matches = result==test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    print accuracy

def main2():

    # Load the data, converters convert the letter to a number
    data= np.loadtxt('letter-recognition.data', dtype= 'float32', delimiter = ',',
                        converters= {0: lambda ch: ord(ch)-ord('A')})

    # split the data to two, 10000 each for train and test
    train, test = np.vsplit(data,2)

    # split trainData and testData to features and responses
    responses, trainData = np.hsplit(train,[1])
    labels, testData = np.hsplit(test,[1])

    # Initiate the kNN, classify, measure accuracy.
    knn = cv2.KNearest()
    knn.train(trainData, responses)
    ret, result, neighbours, dist = knn.find_nearest(testData, k=5)

    correct = np.count_nonzero(result == labels)
    accuracy = correct*100.0/10000
    print accuracy