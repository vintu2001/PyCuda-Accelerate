cmake_minimum_required(VERSION 3.24)
project(PyCudaAccelerate VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(src)

option(BUILD_PYTHON_BINDINGS "Build pybind11 Python bindings" OFF)
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_core python/bindings.cpp)
    target_link_libraries(_core PRIVATE pycuda_algorithms)
    target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    set_target_properties(_core PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )

    install(TARGETS _core
        LIBRARY DESTINATION pycuda_accelerate
        RUNTIME DESTINATION pycuda_accelerate
        ARCHIVE DESTINATION pycuda_accelerate
    )
endif()
