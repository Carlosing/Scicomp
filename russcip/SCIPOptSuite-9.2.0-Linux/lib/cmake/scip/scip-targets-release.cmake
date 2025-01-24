#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "scip" for configuration "Release"
set_property(TARGET scip APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(scip PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/scip"
  )

list(APPEND _cmake_import_check_targets scip )
list(APPEND _cmake_import_check_files_for_scip "${_IMPORT_PREFIX}/bin/scip" )

# Import target "libscip" for configuration "Release"
set_property(TARGET libscip APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libscip PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libscip.so.9.2.0.0"
  IMPORTED_SONAME_RELEASE "libscip.so.9.2"
  )

list(APPEND _cmake_import_check_targets libscip )
list(APPEND _cmake_import_check_files_for_libscip "${_IMPORT_PREFIX}/lib/libscip.so.9.2.0.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
