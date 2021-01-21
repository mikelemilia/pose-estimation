cmake_minimum_required(VERSION 2.8)

project(pose-estimation)
find_package(OpenCV REQUIRED)

set(SOURCE_FILES
        src/main.cpp
        src/ObjectEstimator.cpp
        src/results_writer.cpp
        src/Utility.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

include_directories(${OpenCV_INCLUDE_DIRS})
