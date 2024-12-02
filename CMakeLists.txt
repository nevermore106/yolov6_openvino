cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 11)

project(openvino_test)

set(ov_link_libraries openvino::runtime)
set(PROJECT_ROOT ${CMAKE_SOURCE_DIR})

#模型路径
set(MODEL_PATH "${PROJECT_ROOT}/weight/best_ckpt.xml")
set(MODEL_BIN_PATH "${PROJECT_ROOT}/weight/best_ckpt.bin")
add_definitions(-DMODEL_PATH="${MODEL_PATH}" -DMODEL_BIN_PATH="${MODEL_BIN_PATH}")

#OpenVINO路径
set(OpenVINO_DIR "/home/zxl/Downloads/l_openvino_toolkit_ubuntu20_2023.3.0.13775.ceeafaf64f3_x86_64/runtime/cmake")

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

include_directories(/home/zxl/OpenVINO/include)

add_executable(openvino_test main.cpp number_process.cpp Serial.cpp)

target_link_libraries(openvino_test ${OpenCV_LIBS} ${ov_link_libraries} pthread)

