cmake_minimum_required(VERSION 3.10)

project(rknn_car_recognition_demo)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (ENABLE_ASAN)
    message(STATUS "BUILD WITH ADDRESS SANITIZER")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../RkYoloRtspServer-master/thirdparty/libv4l2cc v4l2cc.out)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/opencv/opencv-android-sdk-build/sdk/native/jni/abi-${CMAKE_ANDROID_ARCH_ABI})
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/opencv/opencv-osx-x86_64/share/OpenCV)
else()
    if (TARGET_SOC STREQUAL "x86-64")
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/opencv/opencv-linux-armhf/share/OpenCV)
    else()
        set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
    endif()
endif()

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV_DIR=${OpenCV_DIR}")
message(STATUS "OpenCV_LIBS=${OpenCV_LIBS}")

if (TARGET_SOC STREQUAL "rv1106" OR TARGET_SOC STREQUAL "rv1103")
    add_definitions(-DRV1106_1103)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/allocator/dma)
endif()

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

set(CAR_RECOGNITION_COMMON_SRCS
    detect.cc
    plate_rec.cc
    car_color.cc
    pipeline.cc
)

add_executable(${PROJECT_NAME}
    main.cc
    ${CAR_RECOGNITION_COMMON_SRCS}
)

add_executable(rknn_car_recognition_stream_demo
    stream_demo.cc
    ${CAR_RECOGNITION_COMMON_SRCS}
)

set(CAR_RECOGNITION_TARGETS
    ${PROJECT_NAME}
    rknn_car_recognition_stream_demo
)

target_include_directories(rknn_car_recognition_stream_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../RkYoloRtspServer-master/thirdparty/libv4l2cc/inc
)
target_link_libraries(rknn_car_recognition_stream_demo v4l2cc)

foreach(target_name ${CAR_RECOGNITION_TARGETS})
    target_link_libraries(${target_name}
        imageutils
        fileutils
        imagedrawing
        ${LIBRKNNRT}
        ${OpenCV_LIBS}
        dl
    )

    if (CMAKE_SYSTEM_NAME STREQUAL "Android")
        target_link_libraries(${target_name} log)
    endif()

    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(THREADS_PREFER_PTHREAD_FLAG ON)
        find_package(Threads REQUIRED)
        target_link_libraries(${target_name} Threads::Threads)
    endif()

    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LIBRKNNRT_INCLUDES}
    )
endforeach()

install(TARGETS ${CAR_RECOGNITION_TARGETS} DESTINATION .)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../model DESTINATION .)
