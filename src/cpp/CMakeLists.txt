set(MY_CURR ${CMAKE_CURRENT_LIST_DIR})

# add_cv_executable(NAME SRCS srcs LIBS libs)
macro(add_cv_executable NAME)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs SRCS LIBS)
  cmake_parse_arguments(THIS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(THIS_SRCS)
    add_executable(${NAME} ${THIS_SRCS})
  else()
    if(EXISTS ${MY_CURR}/${NAME}.cc)
      set(THIS_SRCS ${NAME}.cc)
    elseif(EXISTS ${MY_CURR}/${NAME}.cpp)
      set(THIS_SRCS ${NAME}.cpp)
    else()
      message(FATAL_ERROR "${NAME}.cc|cpp not found")
    endif()
    add_executable(${NAME} ${THIS_SRCS})
  endif()
  target_link_libraries(${NAME} ${OpenCV_LIBS} ${THIS_LIBS})
  target_compile_definitions(${NAME} PUBLIC
    MY_DATA="${MY_DATA}"
  )
endmacro()

add_subdirectory(${MY_CURR}/camera)
add_subdirectory(${MY_CURR}/stitching)
