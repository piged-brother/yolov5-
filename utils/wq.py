import numpy as np

shapes = np.array([[1, 2], [3, 4]])  # 以二维数组形式定义 shapes

s = shapes  # wh
ar = s[:, 1] / s[:, 0]  # aspect ratio计算高宽比
irect = ar.argsort()
print(irect)
shapes = s[irect]  # wh
ar = ar[irect]

print("Sorted Shapes:")
print(shapes)

print(ar)
