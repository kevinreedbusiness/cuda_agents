if(NOT HAS_PYBIND)
  return()
endif()

pybind11_add_module(poe
  bindings.cpp
)

target_link_libraries(poe PRIVATE poe_engine)
