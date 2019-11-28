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

set(cv_names
  camera
  canny
  kinect
)

foreach(cv_name IN LISTS cv_names)
  add_cv_executable(${cv_name})
endforeach()

## install

install(TARGETS ${cv_names}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${MY_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${MY_NAME}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/${MY_NAME}
)
