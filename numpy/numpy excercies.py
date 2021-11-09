'''
Part 1 - basic

Please solve the following exercises. Good Luck!
'''

'''
1. Import the numpy package under the name `np`
'''

import numpy as np

'''
2. Print the numpy version
hint: search on Google "numpy version", the first entries.
'''

print(np.version.version)

'''
3. Create a vector of zeros with the size 10
hint: use np.zeros()
'''

newVector = np.zeros(10)

'''
4. How to find the memory size of any array
hint: itemsize returns the size in memory of each element
'''

print(np.size(newVector))

'''
5. How to get the documentation of the numpy add function from the IPython console?
hint: try "?" before the function name
'''

?np.size

'''
6. Create a vector of zeros with the size 10 but the fifth value which is 1
'''

newVector = np.zeros(10)
newVector[4]=1
print(newVector)

'''
7. Create a vector with values ranging from 10 to 49 (★☆☆)
hint: np.arange()
'''

newVector = np.arange(10, 50, 1) # arguments: start, stop, step
print(newVector)

'''
8. Reverse a vector (first element becomes last) (★☆☆)
hint: [::-1]
'''

reversedVector = newVector[::-1]
print(reversedVector)

'''
9. Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)
hint: reshape
'''

newVector = np.arange(0, 9, 1)
reshapeVector = np.reshape(newVector,(3,3))
print(reshapeVector)

'''
10. Find indices of non-zero elements from \[1,2,0,0,4,0\] (★☆☆)
hint: nonzero
'''

newVector = np.array([1,2,0,0,4,0])
print(np.nonzero(newVector))

'''
11. Create a 3x3 identity matrix (★☆☆)
hint: the best numpy documentation is in the scipy website :)
try https://docs.scipy.org/doc/numpy/reference/generated/numpy.eye.html
'''
newIdentityMatrix = np.eye(3)

'''
12. Create a 3x3x3 array with random values (★☆☆)
hint: Random sampling (numpy.random) — NumPy v1.21 Manual
'''

newRandom3DMatrix = np.random.rand(3,3,3)
print(newRandom3DMatrix)

'''
13. Create a 10x10 array with random values and find the minimum and maximum values
hint: a.min,a.max
'''
newRandom3DMatrix = np.random.rand(10,10)
print(newRandom3DMatrix.max(), newRandom3DMatrix.min())

'''
14. Create a random vector of size 30 and find the mean value
hint: a.mean
'''

newRandomVector = np.random.rand(30)
print(newRandomVector.mean())

'''
15. Create a 2d array with 1 on the border and 0 inside
hint: ones()
hint2: for an array a, which elements are chosen by a[1:-1] ?
'''

newOnesArray = np.ones([5,5], dtype=int)
newOnesArray[1:-1, 1:-1] = 0
print(newOnesArray)

'''
16. How to add a border (filled with 0's) around an existing array?
hint: https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
'''

newOnesArray = np.ones([5,5], dtype=int)
newOnesArray = np.pad(newOnesArray, pad_width=1)
print(newOnesArray)

'''
17. What is the result of the following expression? (★☆☆)
```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
0.3 == 3 * 0.1
```
hint 1: nan is not a number
hint 2: try printing the expressions to see what is printed
'''


print(np.nan == np.nan) # False
print(np.inf > np.nan) # False
print(np.nan - np.nan) # nan
print(0.3 == 3 * 0.1) # False

'''
18. Create a 5x5 matrix with values 1,2,3,4,7 on the diagonal
hint: np.diag
'''

newDiagonalMartix = np.diag([1,2,3,4,7])
print(newDiagonalMartix)

'''
19. Create a 8x8 matrix and fill it with a checkerboard pattern
hint1: slicing an array you can use - list[start:end:step]
hint2: [::2] - for even. [1::2] - for odd
'''

checkerboardMatrix = np.ones([8,8],dtype=int)
for columnNumber in range(0,8):
    if columnNumber%2==0:
        checkerboardMatrix[columnNumber][::2] = 0
    else:
        checkerboardMatrix[columnNumber][1::2] = 0
print(checkerboardMatrix)

'''
20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?
hint: https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.unravel_index.html
'''

print(np.unravel_index(100,(6,7,8)))

'''
21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)
hint: numpy.tile(A, reps)
Construct an array by repeating A the number of times given by reps.
'''

singleLineArray = np.array([1,2,3,4,5,6,7,8])
newCheckerboardMatrix = np.tile(singleLineArray,(8,1))
print(newCheckerboardMatrix)

'''
22. Normalize a 5x5 random matrix
hint: create a random 5*5 matrix
subtract the min value, then divide by (max-min)
'''

newRandomMatrix = np.random.rand(5,5)
updatedRandomMatrix = newRandomMatrix - newRandomMatrix.min()
normalizer = newRandomMatrix.max()-newRandomMatrix.min()
print(updatedRandomMatrix/normalizer)

'''
23. Create an array of 2x4 with dtype numpy.int16. print the dtype of the array
'''

newArray = np.array([1,2,3], dtype=int16)
print(newArray.dtype())

'''
24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)
hint: numpy.dot()
'''

firstMatrix = np.random.rand(5,3)
secondMatrix = np.random.rand(3,2)
print(np.dot(firstMatrix,secondMatrix))
print(firstMatrix@secondMatrix)

'''
25. Given a 1D array, negate all elements which are between 3 and 8, in place.
(hint: >, <=)
'''

randomArray = np.random.uniform(1,10,50)
randomArray = np.int64(randomArray)
print([value for value in randomArray if 3>value and value<8])

'''
26. What is the output of the following script? (★☆☆)
(hint: np.sum)
# Author: Jake VanderPlas
print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
'''

print(sum(range(5),-1)) # 9
from numpy import *
print(sum(range(5),-1)) # 10

'''
27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
'''

Z = np.random.uniform(1,100,100)
print(Z**Z)        # ok
print(2 << Z >> 2) # the inputs could not be safely coerced to any supported types
print(Z <- Z)      # ok, all FALSE
print(1j*Z)        # ok
print(Z/1/1)       # ok
print(Z<Z>Z)       # The truth value of an array with more than one element is ambiguous

'''
28. What are the result of the following expressions?
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
'''

print(np.array(0) / np.array(0))  # nan
print(np.array(0) // np.array(0))  # 0
print(np.array([np.nan]).astype(int).astype(float)) # [-2.14748365e+09]

'''
29. How to round away from zero a float array ? (★☆☆)
(hint: np.uniform, np.copysign, np.ceil, np.abs)
'''

newArray = np.array([1.1,2.2,3.3,4.4,5.5])
absArray = np.ceil(newArray)
print(absArray)

'''
30. How to find common values between two arrays? (★☆☆)
(hint: np.intersect1d)
'''

a1 = np.int64(np.random.uniform(1,100,100))
a2 = np.int64(np.random.uniform(1,100,100))
print(np.intersect1d(a1, a2))

'''
31. How to ignore all numpy warnings (not recommended)? (★☆☆)
(hint: np.seterr, np.errstate)
'''

np.errstate(all='ignore')

'''
32. Is the following expressions true? (★☆☆)
(hint: imaginary number)
np.sqrt(-1) == np.emath.sqrt(-1)
'''

# nope

'''
33. How to get the dates of yesterday, today and tomorrow? (★☆☆)
(hint: np.datetime64, np.timedelta64)
'''

timeData = np.datetime64('today')
print(timeData)
print(timeData + np.timedelta64(1, 'D'))
print(timeData - np.timedelta64(1, 'D'))

'''
34. How to get all the dates corresponding to the month of July 2016? (★★☆)
(hint: np.arange(dtype=datetime64['D']))
'''

print(np.arange('2016-06', '2016-07', dtype='datetime64[D]'))

'''
36. Extract the integer part of a random array using 5 different methods (★★☆)
(hint: %, np.floor, np.ceil, astype, np.trunc)
'''

randomArray1 = np.random.uniform(1,10,5)
print(randomArray1)
randomArray1 = np.int64(randomArray1)
print(randomArray1)

randomArray2 = np.random.uniform(1,10,5)
print(randomArray2)
randomArray2 = np.floor(randomArray2)
print(randomArray2)

randomArray3 = np.random.uniform(1,10,5)
print(randomArray3)
randomArray3 = randomArray3.astype('int64')
print(randomArray3)

randomArray4 = np.random.uniform(1,10,5)
print(randomArray4)
randomArray4 = np.trunc(randomArray4)
print(randomArray4)

randomArray5 = np.random.uniform(1,10,5)
print(randomArray5)
randomArray5 = randomArray5 - randomArray5 % 1
print(randomArray5)
65

'''
37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)
(hint: np.arange)
'''

gradedArray = np.zeros((5,5))
gradedArray += np.arange(5)
print(gradedArray)

'''
39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)
(hint: np.linspace)
'''

gradedVector = np.linspace(0.1,1,num=10,endpoint=False)
print(gradedVector)

'''
40. Create a random vector of size 10 and sort it (★★☆)
(hint: sort)
'''

sortedArray = np.random.uniform(1,10,10)
sortedArray = np.sort(sortedArray)
print(sortedArray)

'''
41. How to sum a small array faster than np.sum? (★★☆)
(hint: np.add.reduce)
'''

newArray = np.random.uniform(1,10,10)
print(np.add.reduce(newArray))

'''
42. Consider two random array A and B, check if they are equal (★★☆)
(hint: np.allclose, np.array_equal)
'''

newArray1 = np.random.uniform(1,10,10)
newArray2 = np.random.uniform(1,10,10)
print(np.array_equal(newArray1,newArray2))
newArray1 = np.zeros((5))
newArray2 = np.zeros((5))
print(np.array_equal(newArray1,newArray2))

'''
43. Make an array immutable (read-only) (★★☆)
(hint: flags.writeable)
'''

newArray = np.random.uniform(1,10,10)
print(newArray.flags.writeable)
newArray.flags.writeable = False
print(newArray.flags.writeable)
newArray[0] = 1  # ValueError: assignment destination is read-only

'''
44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to
polar coordinates (★★☆)
(hint: np.sqrt, np.arctan2)
'''

cartesianArray = np.reshape(np.int64(np.random.uniform(1,10,20)),(10,2))
print(cartesianArray)
polarArray = []
for column in cartesianArray:
    polarArray.append(np.sqrt(np.power(column[0],2)+np.power(column[1],2)))
    polarArray.append(np.arctan2(column[1],column[0]))
polarArray = np.reshape(polarArray,(10,2))
print(polarArray)

'''
45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)
(hint: argmax)
'''

newArray = np.random.uniform(1,10,10)
print(newArray)
newArray[np.argmax(newArray)] = 0
print(newArray)

'''
48. Print the minimum and maximum representable value for each numpy scalar type
(★★☆)
(hint: np.iinfo, np.finfo, eps)
'''

for dtype in [np.int8, np.int32, np.int64]:
    print(dtype)
    print('min: ', np.iinfo(dtype).min)
    print('max: ', np.iinfo(dtype).max, '\n')
for dtype in [np.float32, np.float64]:
    print(dtype)
    print('min: ', np.finfo(dtype).min)
    print('max: ', np.finfo(dtype).max)

'''
50. How to find the closest value (to a given scalar) in a vector? (★★☆)
(hint: argmin)
'''

newArray = np.int64(np.random.uniform(1,50,20))
startingValue = np.int64(np.random.uniform(1,50,1))
nearestValueIndex = (np.abs(newArray-startingValue)).argmin()
print(newArray, startingValue, newArray[nearestValueIndex])

'''
52. Consider a random vector with shape (100,2) representing coordinates, find point by
point distances (★★☆)
(hint: np.atleast_2d, T, np.sqrt, think what is the equation for calculating distances)
'''

randomVector = np.reshape(np.int64(np.random.uniform(1,10,200)),(100,2))
xValues = np.atleast_2d(randomVector[:,0])
xValuesT = np.transpose(xValues)
yValues = np.atleast_2d(randomVector[:,1])
yValuesT = np.transpose(yValues)

distancesArray = []
distancesArray = np.sqrt((xValues-xValuesT)**2 + (yValues-yValuesT)**2)
print(distancesArray)

'''
53. How to convert a float (32 bits) array into an integer (32 bits) in place?
(hint: astype(copy=False))
'''

randomArray = np.random.uniform(1,10,200)
randomArray.astype('float32')
randomArray.astype('int32')

'''
58. Subtract the mean of each row of a matrix (★★☆)
(hint: mean(axis=,keepdims=))
'''

randomMatrix = np.reshape(np.int64(np.random.uniform(1,10,100)),(10,10))

print('sum: ' , np.sum(randomMatrix,axis=1))
print('mean: ' , np.mean(randomMatrix,axis=1))

'''
60. How to tell if a given 2D array has null columns? (★★☆)
(hint: any, ~)
'''

randomMatrix = np.reshape(np.random.uniform(1,10,100),(10,10))
print((~randomMatrix.any(axis=0)).any())
randomMatrix[5][5] = np.nan
randomMatrix[4][4] = 0
print((~randomMatrix.any(axis=0)).any())

'''
64. Consider a given vector, how to add 1 to each element indexed by a second vector (be
careful with repeated indices)? (★★★)
(hint: np.bincount | np.add.at)
'''

newVector = np.random.randint(1,50,20)
print(newVector)

onesVector = np.int64(np.ones(20))
newVector = newVector + onesVector
print(newVector)

'''
67. Considering a four dimensions array, how to get sum over the last two axes at once?
(★★★)
(hint: sum(axis=(-2,-1)))
'''

#new4DArray = np.reshape(np.random.randint(1,50,625),(5,5,5,5))
new4DArray = np.reshape(np.int64(np.ones(625)),(5,5,5,5))
print(new4DArray)

print(np.sum(new4DArray,axis=(-2,-1)))

'''
68. Considering a one-dimensional vector D, how to compute means of subsets of D using a
vector S of same size describing subset indices? (★★★)
(hint: np.bincount)
'''

newVector1 = np.random.randint(1,50,20)
print(newVector1)
newVector2 = np.random.randint(1,50,20)
print(newVector2)
print(np.bincount(newVector2, weights=newVector1) / np.bincount(newVector2))

'''
69. How to get the diagonal of a dot product? (★★★)
(hint: np.diag)
'''

newVector1 = np.random.randint(1,50,(5,5))
newVector2 = np.random.randint(1,50,(5,5))
newVector2Transposed = np.transpose(newVector2)
print(np.sum(newVector1 * newVector2Transposed, axis=1))

'''
70. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros
interleaved between each value? (★★★)
(hint: array[::4])
'''

newVector = np.array([1,2,3,4,5])
updatedVector = []
for i in range(0,len(newVector)-1):
    updatedVector.append(newVector[i])
    for j in range(0,3):
        updatedVector.append(0)
updatedVector.append(newVector[len(newVector)-1])
print(updatedVector)

'''
71. Consider an array of dimension (5,5,3), how to multiply it by an array with dimensions
(5,5)? (★★★)
(hint: array[:, :, None], or np.expand_dims)
'''

new3DArray = np.random.randint(1,50,(3,5,5))
new2DArray = np.random.randint(1,50,(5,5))

multipyOutcome = np.ones((3,5,5))

for i in range(0,3):
    multipyOutcome[:][:][i] = new3DArray[:][:][i] * new2DArray

print(multipyOutcome)

'''
72. How to swap two rows of an array? (★★★)
(hint: array[[]] = array[[]])
'''

newArray = np.random.randint(1,50,(2,5))
print(newArray)
newArray[[1,0]] = newArray[[0,1]]
print(newArray)

'''
75. How to compute averages using a sliding window over an array? (★★★)
(hint: np.cumsum)
'''
newArray = np.random.randint(1,50,(2,5))
print(newArray)
print(np.cumsum(newArray))

'''
77. How to negate a Boolean, or to change the sign of a float in place? (★★★)
(hint: np.logical_not, np.negative)
'''

zeroesOnces = np.random.randint(0,2,10)
print(zeroesOnces)
print(np.logical_not(zeroesOnces, out=zeroesOnces))

flipSignsArray = np.random.uniform(-1.0,1.0,10)
print(flipSignsArray)
print(np.negative(flipSignsArray, out=flipSignsArray))

'''
78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute
distance from p to each line i (P0[i],P1[i])? (★★★)
'''
def distance_from_2_lines_to_point(P0,P1,singlePoint):

    distanceP0 = np.power(P0-singlePoint,2)
    distanceP0sum = np.sum(distanceP0, axis = 1)
    distanceP0sumFinal = np.sqrt(distanceP0sum)
    
    distanceP1 = np.power(P1-singlePoint,2)
    distanceP1sum = np.sum(distanceP1, axis = 1)
    distanceP1sumFinal = np.sqrt(distanceP1sum)
    
    print(distanceP0sumFinal)
    print(distanceP1sumFinal)

P0 = np.random.randint(1,50,(5,2))
P1 = np.random.randint(1,50,(5,2))
singlePoint = np.random.randint(1,50,(1,2))
print('the distances from point ' , singlePoint)
print(distance_from_2_lines_to_point(P0,P1,singlePoint))

'''
79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to
compute distance from each point j (P[j]) to each line i (P0[i],P1[i])? (★★★)
'''

P0 = np.random.randint(1,50,(5,2))
P1 = np.random.randint(1,50,(5,2))
singlePointArray = np.random.randint(1,50,(5,2))

for startingPoint in singlePointArray:
    print('the distances from point ' , startingPoint)
    distance_from_2_lines_to_point(P0,P1,startingPoint)

'''
80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and
centered on a given element (pad with a fill value when necessary) (★★★)
(hint: minimum, maximum)
'''

import random
arbitraryArray = np.random.randint(1,10,(random.randint(1,10),random.randint(1,10)))
fixedShape = (random.randint(1,10),random.randint(1,10))
startingPoint = (random.randint(1,len(arbitraryArray[0])),random.randint(1,len(arbitraryArray[0])))

newArray = np.zeros((fixedShape[0],fixedShape[1]))

i = startingPoint[0]
j = startingPoint[1]

newarrayi = 0
newarrayj = 0

while i < len(arbitraryArray[0]):
    
    while j < len(arbitraryArray[1]):
        
        if newarrayi < fixedShape[0] and newarrayj < fixedShape[1]:
           newArray[newarrayi][newarrayj] = arbitraryArray[i][j]
        
        newarrayj += 1
        j += 1
        
    
    newarrayj = 0
    j = startingPoint[1]
    
    newarrayi += 1
    i += 1

print(newArray)

'''
83. How to find the most frequent value in an array?
(hints: np.unique, return_counts, argmax)
'''

newArray = np.random.randint(1,50,10)
print(np.bincount(newArray).argmax())

'''
89. How to get the n largest values of an array (★★★)
(hint: np.sort)
'''

import random
nValue =  random.randint(1,1000)
newArray = np.random.randint(1,100,1000)
sortedArray = np.sort(newArray)
print(sortedArray[-nValue:])

'''
93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain
elements of each row of B regardless of the order of the elements in B? (★★★)
(hint: np.where)
'''

A = np.random.randint(1,10,(8,3))
B = np.random.randint(1,10,(2,2))

for singleRowA in A:
    for singleRowB in B:
        if singleRowB[0] in singleRowA and singleRowB[1] in singleRowA:
            print(singleRowB, 'was found i ', singleRowA)

'''         
94. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) (★★★)
'''

newMatrix = np.random.randint(1,3,(10,3))
print(newMatrix)
for singleRow in newMatrix:
    if singleRow[0] != singleRow[1] or singleRow[0] != singleRow[2]:
        print(singleRow)
    
'''
96. Given a two dimensional array, how to extract unique rows? (★★★)
(hint: np.ascontiguousarray | np.unique)
'''

twoDimensionalArray = np.random.randint(1,3,(10,3))
print(twoDimensionalArray,'\n')
print(np.unique(twoDimensionalArray, axis=0))

