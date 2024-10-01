cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(MedSAM)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/bowang-lab/MedSAMSlicer")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Reza Asakereh (University Health Network), Jun Ma (University Health Network), Andrew Qiao (University Health Network)")
set(EXTENSION_DESCRIPTION "This module implements MedSAMLite and several of its variant for universal 2D and 3D image segmentation.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/bowang-lab/MedSAMSlicer/master/MedSAM.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/bowang-lab/MedSAMSlicer/master/MedSAMLite/Resources/Icons/Screenshot.png")
set(EXTENSION_DEPENDS "PyTorch") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(MedSAMLite)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
