# Slicer-SAM 2: 3D Slicer Plugin for Segment Anything in Images and Videos

[[`Paper`](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)] [[`BibTeX`](#citing-sam-2)]

![Slicer-SAM 2 Screenshot](assets/slicer_plugin.png?raw=true)

**Segment Anything Model 2.1 (SAM 2.1)** is a foundation model towards solving promptable visual segmentation in images and videos. We have adopted this valuable model developed by [Meta AI](https://ai.meta.com/research/) to detect lesions and various components in 3D medical images.


## Installation

[Installation Guide Video](https://youtu.be/i4h6qCuFbqE)


This code base relies extensively on SAM2.1 original code base. As 3D Slicer python version at the time of development was 3.9 and SAM2.1 requires it to be at least 3.10, the segmentation core and plugin interface should be set up separately.

### Segmentation Backend Setup
Please install SAM 2.1 on a GPU machine with CUDA>=12.4 using:

- Download repository: `git clone https://github.com/bowang-lab/MedSAMSlicer.git`
`cd MedSAMSlicer; git checkout MedSAM2`
- Create virtual environment: `conda create -n medsam2 python=3.12 -y`
- `conda activate medsam2`
- Install [PyTorch](https://pytorch.org/get-started/locally/) 2.4: `conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia`
- `pip install -e .`
- `pip install -r requirements.txt`

You would also need to download checkpoints (various trained SAM2.1 models) and config files to have a working segmentation core. You can do so by running: `./extra_files.sh`

Alternatively, if you are using powershell: `powershell -ExecutionPolicy Bypass -File .\extra_files.ps1`


To ensure the successful installation of the backend, you can run the following script:
```bash
python infer_SAM21_slicer.py --cfg sam2.1_hiera_t.yaml --img_path img_data.npz --gts_path X --propagate N --checkpoint checkpoints/2.1/sam2.1_hiera_tiny --pred_save_dir data/video/segs_tiny
```

### Plugin Setup
1. Install 3D Slicer from its official [website](https://download.slicer.org/). The compatibility of our plugin has been tested with 3D Slicer >= 5.4.0
2. Select the `Welcome to Slicer` drop-down menu in the toolbar at the top and navigate to `Developer Tools > Extension Wizard`.
3. Click on `select Extension` and locate the `SAM2` folder under `MedSAM2/slicer`. Confirm if asked to import new module.
4. Now, from the  `Welcome to Slicer` drop-down menu, under the `Segmentation` sub-menu, `MedSAM2` option is added. By choosing it, you can start using the plugin.


## Getting Started

[Usage Guide Video](https://youtu.be/kt1WN5BciVg)


### Run Backend

You have to run the segmentation core to accept the incoming segmentation requests. You can do it both locally or on a remote computer:
```bash
python server.py
```
This runs the server on the public interface of your device on port 8080.

### Basic Slicer Plugin Usage

1. In 3D Slicer, from the  `Welcome to Slicer` drop-down menu, under the `Segmentation` sub-menu, select `SAM2`.
2. Set IP to the machine running the server. For local machines use 127.0.0.1. Do the same for port.
3. After loading your image, choose the proper preprocessing method from the `Preprocessing` section.
4. In the Red pannel, slide to the beginning of the component you want to segment. From `ROI` section select `Set As Start Slice`.
5. Do similarly to set the end slice as well. The order of these two steps is not important and you can change your selections by selecting a new start/end as many times as you wish. The plugin automatically reposition the Red pannel to the middle of start and end slices.
6. Use `Add Bounding Box` to select as many component you target **in the middle slice**.
7. Now from the `Segmentation` section, `Segment Middle Slice`. After the loading is done, if no segment is visible, you might need to slide the Red pannel view slightly to the left. 
8. You can now refine the inferred segmentation mask using `Refine Middle Slice` and built-in Slicer modules.
9. When you are satisfied with the middle slice, you can propagate the middle mask to other slices by clicking on `Full Segmentation`.  

## License

The models are licensed under the [Apache 2.0 license](./LICENSE). Please refer to our research paper for more details on the models.

<!-- ## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md). -->

<!-- ## Contributors

The SAM 2 project was made possible with the help of many contributors (alphabetical):

Karen Bergan, Daniel Bolya, Alex Bosenberg, Kai Brown, Vispi Cassod, Christopher Chedeau, Ida Cheng, Luc Dahlin, Shoubhik Debnath, Rene Martinez Doehner, Grant Gardner, Sahir Gomez, Rishi Godugu, Baishan Guo, Caleb Ho, Andrew Huang, Somya Jain, Bob Kamma, Amanda Kallet, Jake Kinney, Alexander Kirillov, Shiva Koduvayur, Devansh Kukreja, Robert Kuo, Aohan Lin, Parth Malani, Jitendra Malik, Mallika Malhotra, Miguel Martin, Alexander Miller, Sasha Mitts, William Ngan, George Orlin, Joelle Pineau, Kate Saenko, Rodrick Shepard, Azita Shokrpour, David Soofian, Jonathan Torres, Jenny Truong, Sagar Vaze, Meng Wang, Claudette Ward, Pengchuan Zhang.

Third-party code: we use a GPU-based connected component algorithm adapted from [`cc_torch`](https://github.com/zsef123/Connected_components_PyTorch) (with its license in [`LICENSE_cctorch`](./LICENSE_cctorch)) as an optional post-processing step for the mask predictions. -->

<!-- ## Citing SAM 2

If you use SAM 2 or the SA-V dataset in your research, please use the following BibTeX entry.

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint},
  year={2024}
}
``` -->
