cmake_minimum_required(VERSION 3.5)

project(HPCProject VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Qt
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

# Set project sources
set(PROJECT_SOURCES
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
)

# Create target
if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(HPCProject
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
else()
    if(ANDROID)
        add_library(HPCProject SHARED
            ${PROJECT_SOURCES}
        )
    else()
        add_executable(HPCProject
            ${PROJECT_SOURCES}
        )
    endif()
endif()

# Link Qt libraries
target_link_libraries(HPCProject PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)

# Manually specify MS-MPI's include directory and library
set(MPI_INCLUDE_PATH "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
set(MPI_LIBRARIES "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib")

# Include MS-MPI headers
include_directories(${MPI_INCLUDE_PATH})

# Link MS-MPI libraries
target_link_libraries(HPCProject PRIVATE ${MPI_LIBRARIES})


# Find OpenMP
find_package(OpenMP)

# Add OpenMP flags to the compiler if OpenMP is found
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling support")
    target_link_libraries(HPCProject PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(HPCProject PRIVATE ${OpenMP_CXX_FLAGS})
else()
    message(STATUS "OpenMP not found, disabling support")
endif()

# Set target properties
if(${QT_VERSION} VERSION_LESS 6.1.0)
    set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.example.HPCProject)
endif()

set_target_properties(HPCProject PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

# Install target
include(GNUInstallDirs)
install(TARGETS HPCProject
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Finalize Qt executable for Qt 6
if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(HPCProject)
endif()
