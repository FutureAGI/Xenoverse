import sys
import numpy

n = 10

a = numpy.zeros((n, n))
bm = 1
for i in range(n):
    if(i >= bm):
        a[i, i - bm] = 1.0
    if(i < n - bm):
        a[i, i + bm] = 1.0
    a[i, i] = 0.10

a = a / numpy.sum(a, axis=1, keepdims=True)
for i in range(n):
    a = numpy.matmul(a, a)

print(a)