import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor

def matrix_multiply(a, b):
    return np.dot(a, b)
def sequential_matrix_multiplication(a, b):
    start_time = time.time()
    result = matrix_multiply(a, b)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time
def parallel_matrix_multiplication(a, b, num_workers):
    start_time = time.time()
    chunk_size = len(a) // num_workers
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        chunks = [(a[i:i+chunk_size], b) for i in range(0, len(a), chunk_size)]
        results = list(executor.map(matrix_multiply, *zip(*chunks)))
    result = np.concatenate(results)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

# Example usage
matrix_size = 1000
matrix_a = np.random.rand(matrix_size, matrix_size)
matrix_b = np.random.rand(matrix_size, matrix_size)

# Sequential execution
sequential_result, sequential_time = sequential_matrix_multiplication(matrix_a, matrix_b)
print(f"Sequential Execution Time: {sequential_time} seconds")

# Parallel execution with 4 workers
num_workers = 4
parallel_result, parallel_time = parallel_matrix_multiplication(matrix_a, matrix_b, num_workers)
print(f"Parallel Execution Time with {num_workers} workers: {parallel_time} seconds")

# Ensure the results are the same for sequential and parallel execution
assert np.allclose(sequential_result, parallel_result), "Results do not match!"
