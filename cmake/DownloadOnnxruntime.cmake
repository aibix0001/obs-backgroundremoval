cmake_minimum_required(VERSION 3.16)

# Download ONNX Runtime for Linux x64 with CUDA/TensorRT GPU support
file(REMOVE_RECURSE onnxruntime)

file(
  DOWNLOAD
    https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-gpu-1.23.2.tgz
    onnxruntime-linux-x64-gpu-1.23.2.tgz
  EXPECTED_HASH SHA256=2083e361072a79ce16a90dcd5f5cb3ab92574a82a3ce0ac01e5cfa3158176f53
)
execute_process(
  COMMAND ${CMAKE_COMMAND} -E tar xf onnxruntime-linux-x64-gpu-1.23.2.tgz
)
file(RENAME onnxruntime-linux-x64-gpu-1.23.2 onnxruntime)
execute_process(COMMAND ln -s lib onnxruntime/lib64)

if(EXISTS onnxruntime/lib/cmake/onnxruntime/onnxruntimeTargets.cmake)
  file(READ onnxruntime/lib/cmake/onnxruntime/onnxruntimeTargets.cmake FILE_CONTENT)

  set(OLD_STRING "INTERFACE_INCLUDE_DIRECTORIES \"\${_IMPORT_PREFIX}/include/onnxruntime\"")
  set(NEW_STRING "INTERFACE_INCLUDE_DIRECTORIES \"\${_IMPORT_PREFIX}/include\"")

  string(REPLACE "${OLD_STRING}" "${NEW_STRING}" MODIFIED_CONTENT "${FILE_CONTENT}")
  file(WRITE onnxruntime/lib/cmake/onnxruntime/onnxruntimeTargets.cmake "${MODIFIED_CONTENT}")
endif()
