import concurrent.futures
import tensorflow as tf
import time
import numpy as np
# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    print('GPU is available')
else:
    raise SystemError('No GPU detected')
# Define a GPU-accelerated complex task (matrix multiplication)
def gpu_complex_task(size):
    with tf.device('/device:GPU:0'):
        # Generate random matrices and perform matrix multiplication
        mat1 = tf.constant(np.random.rand(size, size), dtype=tf.float32)
        mat2 = tf.constant(np.random.rand(size, size), dtype=tf.float32)
        result = tf.matmul(mat1, mat2)
        return result.numpy()
# Number of tasks to execute in parallel
num_tasks = 5
task_sizes = [500, 750, 1000, 1250, 1500]  # Example task sizes, you can use different sizes
# Measure the runtime for sequential execution
start_time = time.time()
sequential_results = [gpu_complex_task(size) for size in task_sizes]
sequential_runtime = time.time() - start_time
# Create a ThreadPoolExecutor for parallel execution
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Measure the runtime for parallel execution
    start_time = time.time()
    parallel_results = list(executor.map(gpu_complex_task, task_sizes))
    parallel_runtime = time.time() - start_time
print("Sequential Results:", [result.shape for result in sequential_results])
print("Sequential Runtime:", sequential_runtime, "seconds")
print("Parallel Results:", [result.shape for result in parallel_results])
print("Parallel Runtime:", parallel_runtime, "seconds")
