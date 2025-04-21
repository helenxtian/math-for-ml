import numpy as np

A = np.array([
    [2, 1, 4, 0], 
    [5, 3, 1, 7],
    [8, 0, 2, 6],
    [1, 9, 5, 3],
    [4, 2, 0, 8]
])

# a)
y = np.array([-2, 1, -1, 3, 0])
v_a = np.array([-1, -2, 2, 1])
Av = np.dot(A, v_a)
try:
    c = np.linalg.lstsq(Av.reshape(-1,1), y.reshape(-1,1), rcond=None)[0][0][0]
    residual = np.linalg.norm(c * Av - y)
    print(f"Solution exists with c = {c:.2f}")
except np.linalg.LinAlgError:
    print("No solution exists")

# b) trying k=3 row
v_b = np.array([0, 0, 1, 0, 0])
ans_b = np.dot(v_b, A)
print(ans_b)

# c)
v_c = np.array([0, 2, 5, 0, -4])
ans_c = np.dot(v_c, A)
print(ans_c)

# d) trying k=3 col
v_d = np.array([[0], [0], [1], [0]])
ans_d = np.dot(A, v_d)
print(ans_d)

# e) 
v_e = np.array([-3, -4, 4, 0])
ans_e = np.dot(A, v_e)
print(ans_e)

# f)
a, b = 2, -1
k, j = 2, 0
v = np.zeros(4)
v[k] = a
v[j] = b
ans_f = np.dot(A, v)
print(ans_f)