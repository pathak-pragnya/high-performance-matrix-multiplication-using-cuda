# **High-Performance Matrix Multiplication Using CUDA**

## **Overview**
This project leverages the computational power of GPUs to implement a highly efficient matrix multiplication kernel using the CUDA programming model. The implementation focuses on performance optimization through techniques like shared memory utilization, 2D tiling, thread coarsening, and parameter tuning. The kernel is benchmarked against naive and cuBLAS implementations, with performance analyzed using a roofline model.

---

## **Features**
1. **Optimized CUDA Kernel Implementation**:
   - High-performance matrix multiplication using advanced CUDA features.
   - Incorporates shared memory for faster access and reduced latency.

2. **Performance Tuning**:
   - 2D Tiling for memory coalescing.
   - Shared memory utilization for efficient data access.
   - Thread coarsening to maximize computation per memory access.

3. **Performance Benchmarking**:
   - Evaluated against naive implementation.
   - Benchmarked against NVIDIA cuBLAS for comparative analysis.

4. **Roofline Model Analysis**:
   - Visualization of performance improvements on various input dimensions.

---

## **Technical Details**
### **Techniques Used**
1. **2D Tiling**:
   - Input matrices are divided into smaller tiles that fit into shared memory.
   - Enhances global memory read coalescing and reduces memory latency.

2. **Shared Memory Utilization**:
   - On-chip memory used for temporary data storage within each thread block.
   - Significantly faster than global memory access, enabling high bandwidth utilization.

3. **Thread Coarsening**:
   - Each thread computes multiple output elements instead of just one.
   - Reduces global memory access operations and increases arithmetic intensity.

4. **Parameter Tuning**:
   - Tunable kernel parameters like `BLOCKDIM_X`, `BLOCKDIM_Y`, and tile dimensions (`TILEDIM`) for optimal performance.
   - Constraints ensure correct mapping and efficient resource utilization.

---

## **Performance Evaluation**
### **Results**
1. **Improved Arithmetic Intensity**:
   - Optimizations led to a significant increase in GFLOPs.
   - Enhanced performance by reducing memory access bottlenecks.

2. **Benchmarking**:
   - **Naive Implementation**:
     - Basic implementation with significant memory latency issues.
   - **Optimized Implementation**:
     - Demonstrated substantial speedups.
   - **cuBLAS**:
     - Benchmarked as the gold standard for comparison.

3. **Roofline Model**:
   - Demonstrated how arithmetic intensity and bandwidth utilization improved with optimizations.

---

## **How to Run on Visual Studio**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/pathak-pragnya/cuda-matrix-multiplication.git
   cd cuda-matrix-multiplication

2. Delete the build and out folder
3. Create an empty folder named build
4. Run and configure CMAKE GUI to enable build
5. pen Visual studio code and switch to solution explorer view
6. Go to Quiz8.sln and right click and open
7. Now select "Release" instead of "Debug" from the menubar
8. Right click on quiz from the solution explorer and select build
9. Run the program
