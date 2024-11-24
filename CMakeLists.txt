cmake_minimum_required(VERSION 3.0)
 
project(openvino_test )
 
find_package(OpenCV REQUIRED )
 
# find_package(OpenVINO REQUIRED )
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
set(ov_link_libraries openvino::runtime)
# set(OpenVINO "/home/zihe6/OpenVINO/thirtyparty/runtime/lib/intel64")
# file(COPY test.jpg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
# file(COPY test2.jpg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
# file(COPY my_ckpt.xml DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
# file(COPY my_ckpt.bin DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
include_directories(/home/zihe6/OpenVINO/thirtyparty/runtime/include/openvino)
add_executable(openvino_test main.cpp number_process.cpp AES256.cpp Serial.cpp ATK_MW196.cpp)
message("OpenVINO:${OpenVINO}")
target_link_libraries(openvino_test ${OpenCV_LIBS} ${ov_link_libraries})
