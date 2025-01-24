#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gcg" for configuration "Release"
set_property(TARGET gcg APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gcg PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/gcg"
  )

list(APPEND _cmake_import_check_targets gcg )
list(APPEND _cmake_import_check_files_for_gcg "${_IMPORT_PREFIX}/bin/gcg" )

# Import target "libgcg" for configuration "Release"
set_property(TARGET libgcg APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libgcg PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "libscip"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libgcg.so.3.7.0.0"
  IMPORTED_SONAME_RELEASE "libgcg.so.3.7"
  )

list(APPEND _cmake_import_check_targets libgcg )
list(APPEND _cmake_import_check_files_for_libgcg "${_IMPORT_PREFIX}/lib/libgcg.so.3.7.0.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
