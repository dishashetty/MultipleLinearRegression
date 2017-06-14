# linreg.py
#
# Standalone Python/Spark program to perform linear regression.
# Performs linear regression by computing the summation form of the
# closed form expression for the ordinary least squares estimate of beta.
# 
# TODO: Write this.
# 
# Takes the yx file as input, where on each line y is the first element 
# and the remaining elements constitute the x.
#
# Usage: spark-submit linreg.py <inputdatafile>
# Example usage: spark-submit linreg.py yxlin.csv

# disha karunakar shetty dshetty1@uncc.edu #800966204

import sys
import numpy as np

from pyspark import SparkContext

def X_Xt(tempLine):
 tempLine[0]=1.0
 x=np.array(tempLine).astype('float')
 
 #print tempLine[1]
 return x

def X_Y(tempLine):
 y=np.array(tempLine[0]).astype('float')
 
 #print tempLine[1]
 return y

if __name__ == "__main__":
  if len(sys.argv) !=4:
    print >> sys.stderr, "Usage: linreg <datafile>"
    exit(-1)

  sc = SparkContext(appName="LinearRegression")
  alpha=float(sys.argv[2])
  iteration=int(sys.argv[3])
  

  # Input yx file has y_i as the first element of each line 
  # and the remaining elements constitute x_i
  yxinputFile = sc.textFile(sys.argv[1])

  yxlines = yxinputFile.map(lambda line: line.split(','))
  yxfirstline = yxlines.first()
  yxlength = len(yxfirstline)
  #print "yxlength: ", yxlength

  #dummy floating point array for beta to illustrate desired output format
  beta = np.zeros((2,1))
  
  #getting value of X from function X_Xt
  X =np.asmatrix(yxlines.map(lambda tempLine: (X_Xt(tempLine))).collect()[0])
  
  #getting value of Y from function X_Y
  Y =np.asmatrix(yxlines.map(lambda tempLine: (X_Y(tempLine))).collect()[0])
  
  #calculating linear regression from gradient descent
  for x in range(0, iteration):
      beta1=np.add(beta,alpha*np.dot((X.T),(np.absolute(np.array(Y)-np.array(np.dot(X,beta))))))
  beta=np.array(beta1).tolist()

  
  #
  # Add your code here to compute the array of 
  # linear regression coefficients beta.
  # You may also modify the above code.
  #

  # print the linear regression coefficients in desired output format
  print "beta: "
  for coeff in beta:
      print coeff[0]

  sc.stop()
