add_library(poe_engine
  engine.cpp
  hardware.cpp
)

if(HAS_CUDA)
  target_sources(poe_engine PRIVATE cuda_kernels.cu)
  set_source_files_properties(cuda_kernels.cu PROPERTIES LANGUAGE CUDA)
  target_link_libraries(poe_engine PRIVATE CUDA::cudart)
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(poe_engine PUBLIC OpenMP::OpenMP_CXX)
endif()

target_include_directories(poe_engine PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(poe_cli main.cpp cli.cpp)
target_link_libraries(poe_cli PRIVATE poe_engine)

