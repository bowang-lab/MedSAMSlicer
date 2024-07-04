# MedSAM-Lite 3D Slicer Plugin

This is the official repository for 3D Slicer Plugin for MedSAM: Segment Anything in Medical Images.



https://github.com/bowang-lab/MedSAMSlicer/assets/19947331/c7ef20f1-7e23-4e6c-8be1-e39931eeb841



## Installation

You can watch a video tutorial of installation steps [here](https://youtu.be/qjsTA5WXuS0).

1. Install 3D Slicer from its official [website](https://download.slicer.org/). The compatibility of our plugin has been tested with 3D Slicer >= 5.4.0
2. Download the specific version of the plugin from [releases](https://github.com/bowang-lab/MedSAMSlicer/releases) page and extract to your desired location.
3. In the Slicer App, press Ctrl+4 to open extension manager
4. In "Install Extension" tab lookup PyTorch​ and install the extension. Our plugin currently relies on this. Restart Slicer.
5. Select the `Welcome to Slicer` drop-down menu in the toolbar at the top and navigate to `Developer Tools > Extension Wizard`.
6. Click on `select Extension` and locate the `MedSAM` folder among the extracted files and directories at step 2. Accept if asked to add the new modules.
7. Now, from the  `Welcome to Slicer` drop-down menu, under the `Segmentation` sub-menu, `MedSAMLite` option is added. By choosing it, you get to the final steps.
8. `Choose a folder` to install module dependencies and click on `Install dependencies`. It can take several minutes.
9. Restart 3D Slicer.

## Upgrade

Remove all pre-existing files from both step#2 and step#6 and install the new version as instructed before.


## Usage

You can watch a video guide for usage [here](https://youtu.be/24WtVbljr8g).

1. From the  `Welcome to Slicer` drop-down menu, under the `Segmentation` sub-menu, `MedSAMLite` option is added. By choosing it, you get to the final steps.
2. Load your image file.
3. From `Prepare Data` accordion menu you can choose from preset preprocessing methods (optional, but extremely helpful)
4. From `Select the Region of Interest` accordion menu, you can click on `Attach ROI` to select the region for segmentation. It is possible to use `Set Current Frame As Selection's Start` and `Set Current Frame As Selection's End` to set ROI boundaries on the L-R slice.
5. In `Start Segmentation` accordion menu, you have two different ways to infer the segmentation:
	* If you want to segment multiple regions or a single large one, it is advised to preprocess the whole image first. To do so:
		1. click on `Send Image` button. It will send the whole 3D image to the backend of the module for process. It will take several minutes. You will need to do this step only once for each image. As long as the image is not changed (new preprocessing / cropping / etc.) you do not have to redo this step.
		**Note For Windows Users**. At this step, firewall might stop the backend and ask for the permission. Grant the permission. In some cases you might need to reboot your computer to update permissions. 
		2. Click on `Segmentation` button to get the segmentation results. You can replace your ROI or delete and re-attach it (step 4) as many times as you need and repeat step 5.b without going over step 5.a again.
	* If you need to segment a single smaller region, you can bypass preprocessing and only click on `Single Segmentation`. After confirming, the segmentation would be inferred.
6. In the same section, from the `Segmentation` and `Source volume` drop-down menus you can choose the right segmentation group and manually refine the segmentation results. You can also assign different color to each segmentation mask.
