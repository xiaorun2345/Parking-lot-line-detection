cmake_minimum_required(VERSION 3.6)
project(DetectParkingSlot)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp detectslot.h detectslot.cpp)

add_executable(DetectParkingSlot ${SOURCE_FILES})
find_package(OpenCV REQUIRED)
target_link_libraries( DetectParkingSlot ${OpenCV_LIBS} )
target_link_libraries(DetectParkingSlot ${PROJECT_SOURCE_DIR}/lib/libiniparser.so.1)
