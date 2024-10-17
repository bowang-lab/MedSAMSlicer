cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SAM2)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/SAM2")
set(EXTENSION_CONTRIBUTORS "Reza Asakereh (University Health Network), Sumin Kim (University of Toronto), Jun Ma (University Health Network)")
set(EXTENSION_DESCRIPTION "segment anything2 in 3D medical images")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/SAM2.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/SAM2/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SAM2)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
