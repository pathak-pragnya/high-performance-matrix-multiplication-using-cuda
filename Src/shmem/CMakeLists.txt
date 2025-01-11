# Include directories
include_directories(
  ${SOURCE_DIR}
  ${CUDA_TOOLKIT_INCLUDE}
  ${NVTX_INCLUDE_DIR}
  ${NVML_INCLUDE_DIR}
  ${FREEGLUT_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
  ${CUDA_TOOLKIT_SAMPLES_DIR}/common/inc
)
 
# Locate cuBLAS library
find_library(CUDA_cublas_LIBRARY cublas PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
 
# Collect headers
file(GLOB HEADERS *.h *.cuh)
source_group("Headers" FILES ${HEADERS})
 
# Add CUDA executable
cuda_add_executable(quiz
  ${HEADERS}
  quiz.cu
  OPTIONS --cudart=shared -lineinfo -arch sm_52 -use_fast_math
  DEBUG -Xptxas -v -g -G 
)
 
# Link libraries (including cuBLAS)
target_link_libraries( 
  quiz
  ${CUDA_CUDA_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_CURAND_LIBRARY}
  ${CUDA_cublas_LIBRARY}  # Link cuBLAS
  ${NVML_DEBUG_LIBRARIES}
  ${NVTX_DEBUG_LIBRARIES}
  ${FREEGLUT_DEBUG_LIBRARIES}
  ${GLEW_DEBUG_LIBRARIES}
)
 
target_link_libraries( 
  quiz
  ${CUDA_CUDA_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_CURAND_LIBRARY}
  ${CUDA_cublas_LIBRARY}  # Link cuBLAS
  ${FREEGLUT_RELEASE_LIBRARIES}
  ${GLEW_RELEASE_LIBRARIES}
)