
file(GLOB sources CONFIGURE_DEPENDS *.cu )
foreach(source IN ITEMS ${sources})
    get_filename_component(name "${source}" NAME_WLE)
    add_executable(${name} ${name}.cu)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    endif()
endforeach()
