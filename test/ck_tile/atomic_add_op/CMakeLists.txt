if(NOT SUPPORTED_GPU_TARGETS MATCHES "gfx11") 
    add_gtest_executable(test_atomic test_atomic.cpp)
    set(CTEST_OUTPUT_ON_FAILURE ON)
endif()
