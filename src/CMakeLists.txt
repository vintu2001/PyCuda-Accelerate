add_library(pycuda_algorithms STATIC
    algorithms/sort.cu
    algorithms/reduce.cu
    algorithms/scan.cu
    algorithms/matmul.cu
)

target_include_directories(pycuda_algorithms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pycuda_algorithms PUBLIC CUDA::cudart CUDA::cublas)

set_target_properties(pycuda_algorithms PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
