cmake_minimum_required(VERSION 3.8)

project(pbd_data)

# set(SOURCES 
#     src/main.cpp
# )

# add_executable(${PROJECT_NAME} ${SOURCES})

# target_include_directories(${PROJECT_NAME}
#     PRIVATE 
#         ${PROJECT_SOURCE_DIR}/include
# )

add_executable(data_from_inp src/main_inp.cpp)
target_include_directories(data_from_inp PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(data_from_py src/main_py.cpp)
target_include_directories(data_from_py PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(ex3 src/main_ex3.cpp)
target_include_directories(ex3 PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(ex4_1 src/main_ex4_1.cpp)
target_include_directories(ex4_1 PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(ex4_2 src/main_ex4_2.cpp)
target_include_directories(ex4_2 PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(data_from_py_with_vf src/main_py_vert_fiber.cpp)
target_include_directories(data_from_py_with_vf PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(LV1_fiber_and_bou src/main_LV1.cpp)
target_include_directories(LV1_fiber_and_bou PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(LV1_0718 src/0718.cpp)
target_include_directories(LV1_0718 PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(4w_elems_lv_rv_to_fiber src/main_4w_elems_lv_rv_to_fiber.cpp)
target_include_directories(4w_elems_lv_rv_to_fiber PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(beam src/main_beam.cpp)
target_include_directories(beam PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(4welems_lv_rv src/main_lv_rv_40000elem.cpp)
target_include_directories(4welems_lv_rv PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(whole_heart src/main_whole_heart.cpp)
target_include_directories(whole_heart PRIVATE ${PROJECT_SOURCE_DIR}/include)
