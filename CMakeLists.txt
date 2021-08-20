cmake_minimum_required(VERSION 3.19)

project(examples)

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_BUILD_TYPE EQUAL "Debug")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -O0 -g")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g")
    list(APPEND CMAKE_PREFIX_PATH "./arrayfire/build")
    list(APPEND CMAKE_PREFIX_PATH "./flashlight/build")
else ()
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")
    list(APPEND CMAKE_PREFIX_PATH "./arrayfire/build")
    list(APPEND CMAKE_PREFIX_PATH "./flashlight/build")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing")

find_package(flashlight CONFIG REQUIRED)

add_executable(ac src/actor_critic.cpp)
target_link_libraries(ac flashlight::flashlight)

add_executable(dqn src/dqn.cpp)
target_link_libraries(dqn flashlight::flashlight)

add_executable(ddqn src/ddqn.cpp)
target_link_libraries(ddqn flashlight::flashlight)

add_executable(ppo src/ppo.cpp)
target_link_libraries(ppo flashlight::flashlight)
