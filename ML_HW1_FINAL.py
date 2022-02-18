from math import sqrt
from pickle import FALSE
import numpy             as np
from pytz import InvalidTimeError
import scipy.linalg      as LA
import time
import pandas            as pd
import matplotlib.pyplot as plt
import timeit

# random matrix A. constraints: cols < rows
rows = np.random.randint(6, 9, dtype =int)
cols = np.random.randint(2, 6, dtype = int)
A = np.random.randint(10, size = (rows, cols), dtype = int)
transposeA = np.matrix.transpose(A)
print('A =\n' , A)

# compute eigenvalues of A^TA and AA^T
ATA = np.dot(transposeA, A)
AAT = np.dot(A, transposeA)
eigvalATA = LA.eigvals(ATA)
eigvalAAT = LA.eigvals(AAT)

print('\n ATA: \n' , ATA, '\nEigenvalues of ATA: \n' , eigvalATA , '\n\n AAT: \n' , AAT , '\n Eigenvalues of AAT:\n'  , eigvalAAT)
# ATA and AAT have the same non-zero eigenvalues. the eigenvalues are equal to the reciprocal of the original values

# ATA will always be smaller 
detATA = LA.det(ATA)
print('\ndet(A) = ', detATA)
product =1
for i in eigvalATA:
    product = product * i
print('product of ATA: ' , product)
# observation: the determinant of the smaller matrix is exactly the product of the eigenvalues of that matrix. 

# # ******* 1B. *******
# find eigenvals of inverse from smaller matrix (ATA)
print('******* 1B. *******')
invATA = LA.inv(ATA)
print('Inverse of ATA: \n', invATA)
eigvalInvATA = LA.eigvals(invATA)
print('\nEigenvals of Inverse of ATA\n' , eigvalInvATA)
eigvalATA = LA.eigvals(ATA)
print('\nEigenval of ATA \n' , eigvalATA)
# observation is printed below
print('The relationship between the eigenvalues of the matrix and the eigenvalues of the inverse of the matrix:\nThe eigenvalues of ATA are each equal to the eigenvalues of the inverse of ATA multiplied by 1^-1 (or 1 divided by each eigenvalue). ')
print('\n1 divided by each eigenvalue from the Inverse of ATA: \n')
for i in eigvalInvATA:
    print(1/i)

print('_______')

# ******* 1C. *******
print('******* 1C. *******')
# create matrix C: random, square, non-symmetric 
n = np.random.randint(2,8)
C = np.random.randint(10, size = (n,n), dtype = int)
transC = np.matrix.transpose(C)
while np.array_equal(C, transC): 
    n = np.random.randint(2,8)
    C = np.random.randint(10, size = (n,n), dtype = int)
    transC = np.matrix.transpose(C)
# computer eigenvalues of C and CT
eigvalC = LA.eigvals(C)
eigvalTransC = LA.eigvals(transC)
print('C = \n' , C)
print('eigval of C = \n', eigvalC, '\neigval of C transpose = \n ', eigvalTransC)
# observation: The eigvenvalues of C and C transpose are the same
eigvalC = np.sort(eigvalC)
eigvalTransC = np.sort(eigvalTransC)
print('sum: ' , sum((eigvalC-eigvalTransC)**2))

# ******* 1D. *******
print('******* 1D. *******')
n = np.random.randint(2,6)
D = np.random.randint(10, size = (n,n), dtype = int)
print('D = \n', D, '\n')

# LU Factorization
print('LU Factorization\n')
# LU factorization creates three matrices
# A permutation matrix (P), upper triangular matrix (U), and a lower triangular matrix (L). such that PD = LU
# A = LU, where L is a unit (has ones on the diagonal) lower triangular matrix and U is the upper triangular matrix.
# LU factorization exists if and only if D is a square matrix and each of the leading submatrices has a determinant that is not equal to 0.
# if LU factorization exists for a given matrix, then P^-1*L*U = L*U
[P,L,U] = LA.lu(D)
print('LU:\n', np.dot(L, U))
invP_L = LA.inv(P) @ L
print('The inverse of (P^-1)*U * L should equal LU\n', invP_L @ U)

LU = LA.lu(D)
print('LU Factorization:\n', LU)
print('The following should be equivalent:')
print('PD: \n', LU[0] @ D)
print('UL: \n', LU[1] @ LU[2])

# QR Factorization
print('__________\nQR Factorization\n')
[Q,R]= LA.qr(D)
print('Q:\n', Q,'\nR:\n',R)
# QR Factorization is useful because Q^-1 = Q^`, where Q^` is the complex conjugate transpose of Q. 
# As demonstrated below, the following two matrices should be equivalent
print('Q inverse:\n', LA.inv(Q))
print('Q cc transpose:\n', np.transpose(np.conj(Q)))
# QR Factorization results in two matrices, Q and R (upper triangular matrix)
# if the elements of D are real numbers, then the Q will be orthogonal
# this is useful for both square and non-square matrices
# references: MathWorks documentation page 


# # Singular Value Decomposition (SVD)
print('__________\nSVD\n')
[U,S,VT] = LA.svd(D)
SVD = LA.svd(D)
print(D)

# SVD allows us to decompose the original matrix into three different matrices where D = U*S*(V^T)
# It is a decomposition method to reduce the dimensionality of the data, as SVD represents the original matrix as a linear combination of low-rank matrices (matrices with smaller dimensions than the og)
# U is calculated by the the eigenvectors of AA^T , which make up the columns of U. These columns of U are known as left-singular vectors of D 
# S gives the singular values of the original matrix D. We can construct a diagonal nxn matrix using the np.diag() function
# VT is calculated by the eigenvectors of A^TA . The column vectors of V are known as the right-singular vectors of D
# Similarly to QR Factorization, SVD allows you to decompose a non-square matrix

# with the three matrices, we can reconstruct the original matrix
print('reconstructed:\n', U @ np.diag(S) @ VT)

# SVD allows us to calculate the rank, which is also the number of non-zero singular values of D 
# the rank essentially represents the unique information in the matrix, so the higher the rank value means more unique information
rank = np.linalg.matrix_rank(D)
print('rank: ', rank)
print('Singular Vals of D: ' , S.shape[0])
# references: SVD tutotial from MIT's course BE.400 / 7.548 and SVD for ML from Machine Learning Mastery


# each singular value of S is the square root of the eigenvalues from A^TA
# S are the square root of the eigenvalues calculated from the eigenvectors of A^TA and AA^T

# If D is not square, we want to change the Sigma matrix from nxn into mxn for proper matrix multiplication. 
# If D is a square matrix, then we can just take the diagonal of Sigma
print(S)
Sigma = np.diag(S)
Sigma = np.zeros((D.shape[0], D.shape[1]))
Sigma[:D.shape[1], :D.shape[1]] = np.diag(S)
print(Sigma)


# *******
# Problem 2
# *******
invTime=[]
solveTime = []
matrixSizes = [5,10,20,50,200]

for i in matrixSizes: # for each matrix size 
    for j in range(1000):
        D = np.random.rand(i,i)
        invD = LA.inv(D)
        #LA.solve
        start = timeit.default_timer()
        LA.solve(D, invD)
        stop = timeit.default_timer()
        solveTime.append(stop-start)
        # LA.inv
        start = timeit.default_timer()
        LA.inv(D)
        stop = timeit.default_timer()
        invTime.append(stop-start)
    
# size 5
hist = plt.hist([invTime[0:999], solveTime[0:999]], bins = 20, alpha=0.5, label= ["LA.inv()","LA.solve()"])
plt.title("Histogram of LA.solve and LA.inv time differences for matrix of size 5")
plt.xlabel("Time", fontsize=16)  
plt.ylabel("Frequency", fontsize=16)
plt.legend()
plt.show()
# size 10
hist = plt.hist([invTime[1000:1999], solveTime[1000:1999]], bins = 20, alpha=0.5, label= ["LA.inv()","LA.solve()"])
plt.title("Histogram of LA.solve and LA.inv time differences for matrix of size 10")
plt.xlabel("Time", fontsize=16)  
plt.ylabel("Frequency", fontsize=16)
plt.legend()
plt.show()
# size 20
hist = plt.hist([invTime[2000:2999], solveTime[2000:2999]], bins = 20, alpha=0.5, label= ["LA.inv()","LA.solve()"])
plt.title("Histogram of LA.solve and LA.inv time differences for matrix of size 20")
plt.xlabel("Time", fontsize=16)  
plt.ylabel("Frequency", fontsize=16)
plt.legend()
plt.show()
# size 50
hist = plt.hist([invTime[3000:3999], solveTime[3000:3999]], bins = 20, alpha=0.5, label= ["LA.inv()","LA.solve()"])
plt.title("Histogram of LA.solve and LA.inv Time differences for Matrix of Size 50")
plt.xlabel("Time", fontsize=16)  
plt.ylabel("Frequency", fontsize=16)
plt.legend()
plt.show()
# size 20
hist = plt.hist([invTime[4000:4999], solveTime[4000:4999]], bins = 20, alpha=0.5, label= ["LA.inv()","LA.solve()"])
plt.title("Histogram of LA.solve and LA.inv time differences for matrix of size 200")
plt.xlabel("Time", fontsize=16)  
plt.ylabel("Frequency", fontsize=16)
plt.legend()
plt.show()

