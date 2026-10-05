# Serial CPU vs CUDA GPU Radix Sort

## 1. Project Objective
This project implements and benchmarks a Radix Sort algorithm for 32-bit unsigned integers, comparing a purely serial CPU implementation against a custom parallel CUDA GPU implementation. The goal is to analyze the performance difference and understand parallel scan and scatter algorithms on modern GPU architectures.

## 2. Radix Sort Algorithm
Radix sort is a non-comparative sorting algorithm that sorts data with integer keys by grouping keys by the individual digits which share the same significant position and value. This implementation uses a **1-bit per pass** Least Significant Digit (LSD) approach, meaning it requires exactly 32 passes to sort 32-bit integers.

## 3. CPU Implementation
The CPU implementation performs the 32 passes serially. For each pass, it iterates through the array to count the number of elements with a `0` at the current bit position. It then iterates again to scatter the elements into a secondary buffer based on their bit value, placing `1`s after `0`s.

## 4. CUDA Parallelization Strategy
The CUDA implementation achieves parallelization by replacing the serial counting and scattering steps with parallel primitives, specifically a **Parallel Exclusive Prefix Sum (Scan)**. The process runs entirely on the GPU without CPU intervention during the 32 passes.

## 5. CUDA Kernels
For each of the 32 passes, the following kernels are executed:
1.  **Predicate Kernel**: Evaluates the `b`-th bit of each element. Writes `1` if the bit is `0`, and `0` otherwise.
2.  **Scan Kernels**: A set of kernels performing a hierarchical exclusive prefix sum on the predicate array.
3.  **Compute Total Zeros**: A single-thread kernel that adds the last element's predicate and scan values to determine the total number of elements with a `0` bit.
4.  **Scatter Kernel**: Computes the final destination index for each element and writes it to a secondary buffer.

## 6. Prefix Scan
The exclusive prefix sum is implemented using a **Host-Orchestrated Hierarchical Scan** (Hillis-Steele within blocks). 
- **Local Scan**: Each block computes an exclusive scan using shared memory and outputs the block's total sum to a `block_sums` array.
- **Recursive Scan**: The host recursively calls the scan function on the `block_sums` array until it is small enough to be scanned by a single block.
- **Add**: The scanned `block_sums` are added back to the local scan results to form the global scan.

## 7. Scatter Operation
Using the global scan results, each thread calculates the destination index for its element independently:
- If the bit is `0`: Destination is simply `scan[i]`.
- If the bit is `1`: Destination is `total_zeros + (i - scan[i])`.

## 8. Thread/Block Configuration
The default configuration uses **512 threads per block**. The number of blocks is determined dynamically based on the input size: `(N + 512 - 1) / 512`.

## 9. Correctness Verification
The benchmark framework compares the CPU-sorted array with the CUDA-sorted array element-by-element. It also explicitly checks that the array is strictly non-decreasing (sorted). If a mismatch occurs, the index and values are printed.

## 10. Benchmark Methodology
- A warm-up execution of 1000 elements is performed before benchmarking to account for CUDA initialization overhead.
- For each size (100k to 10M), the CPU algorithm is timed once.
- The CUDA algorithm is timed multiple times (default 3) and the average is reported.
- Same deterministic random data is used for both CPU and CUDA implementations.

## 11. Timing Methodology
- **CPU Time**: Measured using `std::chrono::high_resolution_clock`.
- **Kernel Time**: Measured using `cudaEventElapsedTime` around the 32 radix passes, excluding Host-Device memory transfers.
- **Total Time**: Measured using `cudaEventElapsedTime` including `cudaMemcpy` (Host to Device), the 32 radix passes, and `cudaMemcpy` (Device to Host).

## 12. Speedup Formulas
- **Kernel Speedup** = CPU Time / Kernel Time
- **End-to-End Speedup** = CPU Time / Total CUDA Time

## 13. How to Compile
Ensure you have `nvcc` and `g++` installed.
```bash
make
```

## 14. How to Run
To run the small correctness test:
```bash
./radix_sort
```
To run the full benchmark suite:
```bash
./radix_sort --benchmark
```

## 15. Expected Output Format
The output will display the CUDA GPU properties followed by a table of results, and save the data to `results/benchmark_results.csv`:
```
+------------+------------+-------------+-------------+-------------+----------------+
| Input Size | CPU (ms)   | Kernel (ms) | Total (ms)  | K-Speedup   | E2E Speedup    |
+------------+------------+-------------+-------------+-------------+----------------+
| 100000     | ...        | ...         | ...         | ...         | ...            |
...
```
