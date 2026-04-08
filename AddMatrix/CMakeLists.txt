cmake_minimum_required(VERSION 3.21)
project(AddMatrix LANGUAGES HIP)

add_executable(${PROJECT_NAME} AddMatrix.hip)
target_link_libraries(${PROJECT_NAME} rocrand)
