cmake_minimum_required(VERSION 3.21)
project(QueryDevice LANGUAGES HIP)

add_executable(${PROJECT_NAME} QueryDevice.hip)
target_link_libraries(${PROJECT_NAME} rocrand)
