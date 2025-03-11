find_package(CUDAToolkit)

file(GLOB sources CONFIGURE_DEPENDS *.cu )
foreach(source IN ITEMS ${sources})
    get_filename_component(name "${source}" NAME_WLE)
    add_executable(${name} ${name}.cu)
endforeach()
