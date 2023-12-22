#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 00:03:26 2023

@author: nici
"""

from time import time


# (1) MATRIX MULTIPLICATION

d = 50

A1 = np.random.normal(size=[d, d])
x1 = np.random.normal(size=[d, 2])

A2 = A1.copy()
x2 = x1.copy()

A2 = A1.reshape(d, d, 1)
x3 = x1.reshape(d, 2, 1)

# np matmul
start = time()
for i in range(10000):
    x1 = np.matmul(A1, x1)
    x1 = sigmoid(x1)
end = time()
print("MatMul:", round(end-start, 3), "s")
print("x:", x1[0:3].flatten())

# Einstein sum
start = time()
for i in range(10000):
    x2 = np.einsum('AB,BC->AC', A1, x2)
    x2 = sigmoid(x2)
end = time()
print("EinSum:", round(end-start, 3), "s")
print("x:", x2[0:3].flatten())

# Einstein sum with extra unnecessary dimension
start = time()
for i in range(10000):
    x3 = np.einsum('ABK,BCK->ACK', A2, x3)
    x3 = sigmoid(x3)
end = time()
print("EinSum:", round(end-start, 3), "s")
print("x:", x3[0:3].flatten())


# (2) HADAMARD PRODUCT

# Standard Hadamard product
start = time()
for i in range(100000):
    had = A1*A2
end = time()
print("MatProd:", round(end-start, 3), "s")
print(had)


# EinSum Hadamard product
start = time()
for i in range(100000):
    had = np.einsum('AB,AB->AB', A1, A2)
print("EinSum:", round(end-start, 3), "s")
print(had)


# (3) ROW/COLUMN SUM

# Standard row sum
start = time()
for i in range(100000):
    S = np.sum(A1, axis=1)
end = time()
print("MatProd:", round(end-start, 3), "s")
print(S)

# EinSum row sum
start = time()
for i in range(100000):
    S = np.einsum('AB->A', A1)
print("EinSum:", round(end-start, 3), "s")
print(S)


# (4) Cauchy vs. Student t with df=1

X = np.random.normal(size=10000000)

start = time()
stats.t(loc=0, scale=1, df=1).logpdf(X)
end = time()
print("Student:", round(end-start, 3))

start = time()
stats.cauchy(loc=0, scale=1).logpdf(X)
end = time()
print("Cauchy:", round(end-start, 3))

start = time()
F_innov(loc=0, scale=1, df=1.).logpdf(X)
end = time()
print("F_innov:", round(end-start, 3))