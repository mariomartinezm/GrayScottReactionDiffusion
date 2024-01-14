cmake_minimum_required(VERSION 3.25)

project(GrayScottReactionDiffusion LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE
    main.cpp)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)

configure_file(compute_shader.glsl  compute_shader.glsl  COPYONLY)
configure_file(vertex_shader.glsl   vertex_shader.glsl   COPYONLY)
configure_file(fragment_shader.glsl fragment_shader.glsl COPYONLY)

set(SIMGLL_REPO_URL "https://github.com/mariomartinezm/simgll")
set(SIMGLL_REPO_BRANCH "master")

FetchContent_Declare(
    simgll
    GIT_REPOSITORY ${SIMGLL_REPO_URL}
    GIT_TAG ${SIMGLL_REPO_BRANCH}
)

FetchContent_MakeAvailable(simgll)

if(UNIX)
    target_link_libraries(${PROJECT_NAME} GL)
    target_link_libraries(${PROJECT_NAME} GLEW)
    target_link_libraries(${PROJECT_NAME} glfw)
    target_link_libraries(${PROJECT_NAME} simgll)
endif(UNIX)

if(WIN32)
    target_include_directories(${PROJECT_NAME}
        PRIVATE ${CMAKE_PREFIX_PATH}/include)

    target_link_libraries(${PROJECT_NAME} opengl32)
    find_library(GLEW_LIB glew32)
    target_link_libraries(${PROJECT_NAME} ${GLEW_LIB})
    find_library(GLFW_LIB glfw3dll)
    target_link_libraries(${PROJECT_NAME} ${GLFW_LIB})
    target_link_libraries(${PROJECT_NAME} simgll)
endif(WIN32)
