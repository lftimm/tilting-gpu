cmake_minimum_required(VERSION 3.21)
project(DotProduct LANGUAGES HIP)

add_executable(${PROJECT_NAME} DotProduct.hip)
target_link_libraries(${PROJECT_NAME} rocrand)
