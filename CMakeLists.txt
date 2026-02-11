cmake_minimum_required(VERSION 3.16)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/common/bootstrap.cmake" NO_POLICY_SCOPE)

project(${_name} VERSION ${_version})

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES "75;86;89")

option(ENABLE_FRONTEND_API "Use obs-frontend-api for UI functionality" OFF)
option(ENABLE_QT "Use Qt functionality" OFF)
option(ENABLE_NVTX_PROFILING "Enable NVIDIA Nsight NVTX profiling markers" OFF)

set(VCPKG_TARGET_TRIPLET "" CACHE STRING "Vcpkg target triplet to use")
option(USE_PKGCONFIG "Use pkg-config to find dependencies" ON)
option(USE_SYSTEM_ONNXRUNTIME "Use system ONNX Runtime" ON)

include(compilerconfig)
include(defaults)
include(helpers)

add_library(${CMAKE_PROJECT_NAME} MODULE)

find_package(libobs REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::libobs)

if(ENABLE_FRONTEND_API)
  find_package(obs-frontend-api REQUIRED)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::obs-frontend-api)
endif()

if(ENABLE_QT)
  find_package(Qt6 COMPONENTS Widgets Core)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE Qt6::Core Qt6::Widgets)
  set_target_properties(
    ${CMAKE_PROJECT_NAME}
    PROPERTIES AUTOMOC ON AUTOUIC ON AUTORCC ON
  )
endif()

# CUDA Runtime (required for GPU detection and preprocessing)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE CUDA::cudart)

# NVTX profiling support (optional, for Nsight Systems/Compute)
if(ENABLE_NVTX_PROFILING)
  message(STATUS "NVTX profiling enabled - link with Nsight Systems for analysis")
  target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE ENABLE_NVTX_PROFILING)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE CUDA::nvToolsExt)
endif()

if(NOT VCPKG_TARGET_TRIPLET STREQUAL "")
  # Use vcpkg to find dependencies
  message(STATUS "Using vcpkg with target triplet: ${VCPKG_TARGET_TRIPLET}")

  list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}")

  find_package(CURL CONFIG REQUIRED)
  find_package(OpenCV CONFIG REQUIRED)

  add_library(OpenCV::opencv_core ALIAS opencv_core)
  add_library(OpenCV::opencv_imgproc ALIAS opencv_imgproc)
elseif(USE_PKGCONFIG)
  # Use pkg-config to find dependencies (Arch Linux, etc.)
  message(STATUS "Using pkg-config to find dependencies")

  find_package(PkgConfig REQUIRED)

  # CURL
  pkg_check_modules(PC_CURL REQUIRED libcurl)
  add_library(CURL::libcurl INTERFACE IMPORTED)
  target_link_libraries(CURL::libcurl INTERFACE ${PC_CURL_LIBRARIES})
  target_include_directories(CURL::libcurl INTERFACE ${PC_CURL_INCLUDE_DIRS})
  target_compile_definitions(CURL::libcurl INTERFACE ${PC_CURL_CFLAGS_OTHER})
  target_link_directories(CURL::libcurl INTERFACE ${PC_CURL_LIBRARY_DIRS})

  # OpenCV
  pkg_check_modules(PC_OPENCV REQUIRED opencv4)
  add_library(OpenCV::opencv_core INTERFACE IMPORTED)
  target_link_libraries(OpenCV::opencv_core INTERFACE opencv_core)
  target_include_directories(OpenCV::opencv_core INTERFACE ${PC_OPENCV_INCLUDE_DIRS})
  target_compile_definitions(OpenCV::opencv_core INTERFACE ${PC_OPENCV_CFLAGS_OTHER})
  target_link_directories(OpenCV::opencv_core INTERFACE ${PC_OPENCV_LIBRARY_DIRS})
  add_library(OpenCV::opencv_imgproc INTERFACE IMPORTED)
  target_link_libraries(OpenCV::opencv_imgproc INTERFACE opencv_imgproc)
  target_link_directories(OpenCV::opencv_imgproc INTERFACE ${PC_OPENCV_LIBRARY_DIRS})
else()
  # Let system find dependencies (Fallback)
  message(STATUS "Using system to find dependencies")

  find_package(CURL REQUIRED)
  find_package(OpenCV CONFIG REQUIRED)
endif()

# ONNX Runtime - Linux only (GPU build with CUDA + TensorRT)
if(NOT USE_SYSTEM_ONNXRUNTIME)
  set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime")
  list(PREPEND CMAKE_PREFIX_PATH "${ONNXRUNTIME_DIR}")
endif()

find_package(onnxruntime CONFIG REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE onnxruntime::onnxruntime)

if(NOT USE_SYSTEM_ONNXRUNTIME)
  file(GLOB ONNXRUNTIME_SO_FILES "${ONNXRUNTIME_DIR}/lib64/libonnxruntime.so.*")
  file(GLOB ONNXRUNTIME_PROVIDER_FILES "${ONNXRUNTIME_DIR}/lib64/libonnxruntime_providers_*.so")
  install(
    FILES ${ONNXRUNTIME_SO_FILES} ${ONNXRUNTIME_PROVIDER_FILES}
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/obs-plugins/${CMAKE_PROJECT_NAME}
  )
  set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/${CMAKE_PROJECT_NAME}")
endif()

add_subdirectory(src/update-checker/CurlClient)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE CurlClient OpenCV::opencv_core OpenCV::opencv_imgproc)

# NVIDIA GPU execution providers are always available (CUDA + TensorRT)
message(STATUS "NVIDIA GPU execution providers enabled (CUDA + TensorRT)")

target_sources(
  ${CMAKE_PROJECT_NAME}
  PRIVATE
    src/plugin-main.c
    src/ort-utils/ort-session-utils.cpp
    src/ort-utils/gpu-info.cpp
    src/ort-utils/async-inference-queue.cpp
    src/ort-utils/cuda-preprocess.cu
    src/obs-utils/obs-utils.cpp
    src/obs-utils/obs-config-utils.cpp
    src/update-checker/github-utils.cpp
    src/update-checker/update-checker.cpp
    src/background-filter-info.c
    src/background-filter.cpp
    src/enhance-filter.cpp
    src/enhance-filter-info.c
)

set_target_properties_plugin(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME ${_name})
