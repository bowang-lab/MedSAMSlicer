# Lite-MedSAM
This is the official repository for MedSAM: Segment Anything in Medical Images.


## Installation
1. Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it `conda activate medsam`
2. `pip install torch==2.0.1 torchvision==0.15.2`
3. Enter the MedSAM folder `cd MedSAM` and run `pip install -e .`


## Get Started

### 3D image segmentation (3D bounding box)

```bash
python medsam_lite_infer_3Dbox.py -i ./HCC_004_0000.nii.gz -o ./

```



### 2D image segmentation (2D bounding box)



```bash
jupyter lab
```

run `medsam_lite_2d_img`