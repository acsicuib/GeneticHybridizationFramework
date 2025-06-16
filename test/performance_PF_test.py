import numpy as np
import time
from resopt.files.arrange import get_pareto_front_from_array

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


def original_get_pareto_front_from_array(array):
    """Original implementation for comparison"""
    size  = array.shape[0]
    n_obj = array.shape[1]
    array_assigned = [True] * size

    for i in range(size):
        if not array_assigned[i]:
            continue

        for j in range(size):
            if not array_assigned[j] or i == j:
                continue

            all_o = True
            for obj in range(n_obj):
                if array[i,obj] > array[j,obj]:
                    all_o = False
                    break

            if all_o: array_assigned[j] = False

    return array[array_assigned]

def performance_test():
    """Compare performance of original vs improved algorithm"""
    
    # Test with different array sizes
    sizes = [100, 500, 1000, 2000]
    n_objectives = 3
    
    print("Performance Comparison: Original vs Improved Algorithm")
    print("=" * 60)
    print(f"{'Size':<8} {'Original (s)':<15} {'Improved (s)':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for size in sizes:
        # Generate random test data
        np.random.seed(42)  # For reproducible results
        array = np.random.rand(size, n_objectives) * 10
        
        # Test original algorithm
        start_time = time.time()
        result_original = original_get_pareto_front_from_array(array)
        original_time = time.time() - start_time
        
        # Test improved algorithm
        start_time = time.time()
        result_improved = get_pareto_front_from_array(array)
        improved_time = time.time() - start_time
        
        # Test chatpgt algorithm
        start_time = time.time()
        result_gpt = pareto_front(array)
        gpt_time = time.time() - start_time

        # Verify results are the same
        if len(result_original) != len(result_improved):
            print(f"WARNING: Results differ for size {size}!")
        else:
            # Sort both results for comparison
            result_original_sorted = result_original[np.lexsort(result_original.T)]
            result_improved_sorted = result_improved[np.lexsort(result_improved.T)]
            result_gpt_sorted = result_gpt[np.lexsort(result_gpt.T)]
            
            if not np.array_equal(result_original_sorted, result_improved_sorted):
                print(f"WARNING: Results differ for size {size}!")
        
        speedup = original_time / improved_time if improved_time > 0 else float('inf')
        speedup2 = original_time / gpt_time if improved_time > 0 else float('inf')
        speedup3 = improved_time / gpt_time if improved_time > 0 else float('inf')
        
        print(f"{size:<8} {original_time:<15.4f} {improved_time:<15.4f} {gpt_time:<15.4f} {speedup:<10.2f}x {speedup2:<10.2f}x {speedup3:<10.2f}x")

if __name__ == "__main__":
    performance_test() 