cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(MedSAM)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/MedSAM")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Reza Asakereh (University Health Network), Andrew Qiao (University Health Network), Anonymous")
set(EXTENSION_DESCRIPTION "3D Slicer extension for MedSAM.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/MedSAM.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/MedSAM/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(t_SegmentData)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
