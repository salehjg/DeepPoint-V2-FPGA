include_directories(
        ${CMAKE_SOURCE_DIR}/inc
        ${CMAKE_SOURCE_DIR}/test/common
        )

file(GLOB_RECURSE TEST_SOURCES
        ${CMAKE_SOURCE_DIR}/test/unittests/*.cpp
        )
message("TEST_SOURCES: ")
message("${TEST_SOURCES}")

set(SOURCES
        ${CMAKE_SOURCE_DIR}/src/CTensorBase.cpp
        ${TEST_SOURCES}
        )
message("SOURCES: ")
message("${SOURCES}")

add_executable(tests_main
        ${CMAKE_SOURCE_DIR}/test/ocltests/tests_main.cpp
        ${SOURCES}
        )

add_test(NAME tests_main COMMAND tests_main)
target_link_libraries(tests_main gtest ${SDAccel_LIBRARIES} ${SDAccel_FLOATING_POINT_LIBRARY} z stdc++fs spdlog)