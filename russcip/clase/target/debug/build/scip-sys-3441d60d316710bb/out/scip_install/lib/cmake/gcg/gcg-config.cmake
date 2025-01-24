if(NOT TARGET libgcg)
  include("${CMAKE_CURRENT_LIST_DIR}/gcg-targets.cmake")
endif()

if(0)
   set(SCIP_DIR "/__w/scipoptsuite-deploy/scipoptsuite-deploy/scip_install/lib64/cmake/scip")
   find_package(SCIP QUIET CONFIG)
endif()

set(GCG_LIBRARIES libgcg)
set(GCG_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/../../../include")
set(GCG_FOUND TRUE)
