import numpy as np

mat = np.array([
    [4, 5, 3], 
    [6, 6, 3],
    [4, 7, 1],
    [3, 8, 2]
])
eco_index = np.array([[4], [2], [6]])
weights = np.array([0.1, 0.2, 0.3, 0.4])


print(mat)

annual_impact = np.dot(mat,eco_index)
print(annual_impact)

weighted_avg = np.dot(weights, mat)
print(weighted_avg)

weighted_total_impact = np.dot(weights, annual_impact)
print(weighted_total_impact)