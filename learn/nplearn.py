import numpy as np

a = np.array([2, 2, 23, 4], dtype=np.float)
print(a.dtype)

b = np.array([[1, 2, 3],
              [2, 4, 6]])

c = np.zeros((3, 4))
c = np.empty((3, 4))
c = np.arange(10, 34, 2).reshape((3, 4))  #左包右不包
c = np.linspace(1, 10, 5)  # 分割5段

print(c)

# numpy 基础运算
print('_____________')
a = np.array([10, 20, 30, 40])
b = np.arange(4)
c = a + b
c = b ** 2
c = 10 * np.sin(a)
a = np.array([[1, 1],
              [0, 1]])
b = np.arange(4).reshape((2, 2))
c = a * b
c_dot = np.dot(a, b)
print(a)
print(b)
print(c)
print(c_dot)

ar = np.random.random((2, 4))
print(ar)
print(np.sum(ar))
print(np.min(ar))
print(np.max(ar, axis=0))

# numpy 基础运算2
print('——————————————————————————————————————————————————')

A = np.arange(2, 14).reshape((3, 4))
print(A)
print(np.argmin(A))  # 最小值索引
print(np.argmax(A))
print(np.mean(A))
print(np.average(A))
print(np.cumsum(A))  # 数组，累积前面的值
print(np.diff(A))  # 数组 差值
print(np.nonzero(A))
print(np.sort(A.T))
print(np.transpose(A))
print(A.T.dot(A))
print(np.clip(A, 4, 10))  # 替换小于最小值的和大宇最大值的
print(np.mean(A, axis=0))

# numpy 索引
print('-----------------------------------------------')
A = np.arange(3, 15).reshape((3, 4))
print(A)
print(A[1])
print(A[1, 1])  # A[1][1] 等价
print(A[:, 1])
print(A[1, 1:3])
for row in A:
    print(row)
for column in A.T:  # 对列迭代
    print(column)

for item in A.flat:  # 对每一项迭代
    print(item)
# numpy 的array 合并
print('-----------------------------------------')
A = np.array([1, 1, 1])[:, np.newaxis]
B = np.array([2, 2, 2])[:, np.newaxis]
C = np.vstack((A, B))
D = np.hstack((A, B))  # 左右合并
print(np.vstack((A, B)))  # 上下合并
print(A.shape, C.shape)
print(D)
# print(A[np.newaxis, :].T)  # 在行上增加一个维度，数组变矩阵
C = np.concatenate((A, B, B, A), axis=1)
print(C)

# numpy array的分割
print('--------------------------------------------------')
A = np.arange(12).reshape((3, 4))
print(np.split(A, 2, axis=1))  # 竖向分割
print(np.array_split(A, 3, axis=1))  # 用array_split做不等量的分割
print(np.vsplit(A, 3))
print(np.hsplit(A, 2))  # 此种方法也只能做等量分割

# numpy array数组拷贝 和 深拷贝
print('---------------------------------------------------')

a = np.arange(4)
b = a
c = a
d = b
a[0] = 11
b is a  # 指针指向同一个对象
b = a.copy()  # deep copy
print(b is a)

