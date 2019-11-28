set(MY_CURR ${CMAKE_CURRENT_LIST_DIR})

get_filename_component(MY_NAME ${MY_CURR} NAME)
set(_outdir "${MY_ROOT}/_output")
set_outdir(
  ARCHIVE "${_outdir}/lib/${MY_NAME}"
  LIBRARY "${_outdir}/lib/${MY_NAME}"
  RUNTIME "${_outdir}/bin/${MY_NAME}"
)
unset(_outdir)

## targets

add_cv_executable(tutorial_stitching
  SRCS tutorial/stitching.cpp
)
add_cv_executable(tutorial_stitching_detailed
  SRCS tutorial/stitching_detailed.cpp
)

## install

install(TARGETS tutorial_stitching tutorial_stitching_detailed
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${MY_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${MY_NAME}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/${MY_NAME}
)