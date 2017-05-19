cmake_minimum_required(VERSION 2.8.9)
project(randomPolicy)

if(COMMAND cmake_policy)
 cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

find_package(Pandora REQUIRED)
include_directories(${PANDORA_INCLUDE_DIRS})

find_package(Boost COMPONENTS filesystem system date_time timer chrono REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

find_package(PythonInterp 2.7 REQUIRED)

set(generateMPI ${Pandora_DIR}/../bin/generateMpi.py)

set(mpiAgentsSrc mpiCode/FactoryCode.cxx mpiCode/RandomAgent_mpi.cxx)
set(mpiAgentsSrcFullPath ${CMAKE_CURRENT_SOURCE_DIR}/mpiCode/FactoryCode.cxx ${CMAKE_CURRENT_SOURCE_DIR}/mpiCode/RandomAgent_mpi.cxx)
set(AgentsSrc 'main.cxx' 'RandomAgent.cxx')
set(AgentsEnv '{"namespaces":["Examples"]}')

MESSAGE( STATUS "AgentsEnv:         " ${AgentsEnv} )
MESSAGE( STATUS "AgentsSrc:         " ${mpiAgentsSrc} )

add_custom_command(
  OUTPUT ${mpiAgentsSrcFullPath}
  COMMAND ${PYTHON_EXECUTABLE} ${generateMPI} --target ${mpiAgentsSrc} --source ${AgentsSrc} --env ${AgentsEnv}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mpiCode)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/mpiCode)

file(GLOB SOURCES "*.cxx")
file(COPY config.xml DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(randomPolicy ${SOURCES} ${mpiAgentsSrcFullPath})
target_link_libraries(randomPolicy pandora ${Boost_LIBRARIES} ${TinyXML_LIBRARIES})
#install(TARGETS randomWalk DESTINATION ${INSTALL_LIB_DIR})
