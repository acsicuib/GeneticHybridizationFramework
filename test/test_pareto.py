import numpy as np
from resopt.files.arrange import get_pareto_front_from_array
from test.performance_PF_test import original_get_pareto_front_from_array

def is_dominated(a, b):
    return np.all(a >= b) and np.any(a > b)

def pareto_front(points):
    pareto = []
    for i, point in enumerate(points):
        dominated = False
        for j, other in enumerate(points):
            if i != j and is_dominated(point, other):
                dominated = True
                break
        if not dominated:
            pareto.append(point)
    return np.array(pareto)



# Test with the same array from the original code
array = np.array([[3,3,5], [3,3,4],[2,3,4],[4,3,4],[3,2,4],[3,4,4], 
                  [3,1,3],[2,2,3],[3,2,3],[4,2,3],[1,3,3],[2,3,3],
                  [3,3,3],[4,3,3],[5,3,3],[2,4,3],[3,4,3],[4,4,3],
                  [3,5,3], [3,2,2],[2,3,2],[3,3,2],[4,3,2],[3,4,2], [3,3,1]])

print("Original array shape:", array.shape)
print("Original array:")
print(array)

result = get_pareto_front_from_array(array)
print("\nPareto front shape:", result.shape)
print("Pareto front:")
print(result)

print("ORIGINAL Pareto front:", original_get_pareto_front_from_array(array)) 


print("GPT Pareto front:", pareto_front(array)) 


# Test with a simple 2D case
simple_array = np.array([[1, 2], [2, 1], [3, 3], [1, 1], [2, 2]])
print("\n\nSimple 2D test:")
print("Input:", simple_array)
print("Pareto front:", get_pareto_front_from_array(simple_array)) 
print("ORIGINAL Pareto front:", original_get_pareto_front_from_array(simple_array)) 
