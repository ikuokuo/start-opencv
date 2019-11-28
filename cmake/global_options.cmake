
option(WITH_OPENNI "With OpenNI" ON)
if(WITH_OPENNI)
  add_definitions(-DWITH_OPENNI)
endif()

# summary

status("")
status("Options:")
status("  WITH_OPENNI: ${WITH_OPENNI}")
