<img src= 'https://github.com/uf-aiaos/ShinyAnimalCV/blob/main/ExampleData/Images_for_demonstration/hexsticker.png' height="200" align="right"/>
<h1 align="center">ShinyAnimalCV</h1>


<p align="center">
Open-source cloud-based web application for object detection, segmentation, and three-dimensional visualization of animals using computer vision
</p>
  
<!-- TABLE OF CONTENTS -->
<details open>
<summary>Table of Contents</summary>

- [Introduction](#introduction)
- [Section 1: Video demos for online usage of ShinyAnimalCV](#section-1)
  - [1.1: Object detection and segmentation](#subsection-1-1)
  - [1.2: Three-dimensional morphological feature extraction and visualization](#subsection-1-2)
- [Section 2: Training computer vision models with custom data](#section-2)
  - [Subsection 2.1: Image annotation](#section-2-1)
  - [Subsection 2.2:  Model training and evaluation](#section-2-2)
- [Section 3: Local inference with trained models in ShinyAnimalCV](#section-3)
- [Contact information and help](#section-contact)
- [References](#section-reference)
- [License](#section-license)
</details>

<!-- Introduction -->
## Introduction<a name="introduction"></a>

Computer vision (CV), a non-intrusive and cost-effective technology, has facilitated the development of precision livestock farming by enabling optimized decision-making through timely and individualized animal care. However, despite the availability of various CV tools in the public domain, applying these tools to animal data can be complex, often requiring users to have programming and data analysis skills, as well as access to computing resources such as CPUs and GPUs. Moreover, the rapid expansion of precision livestock farming is creating a growing need to educate and train animal science students about CV. Thus, we developed a user-friendly online web application named `ShinyAnimalCV` for object detection and segmentation, three-dimensional visualization, as well as 2D and 3D morphological feature extraction using animal image data. 

This documentation aims to offer users comprehensive and step-by-step instructions for utilizing `ShinyAnimalCV`, both online and locally. 
[The first section](#section-1) provides users with video demonstrations on how to utilize `ShinyAnimalCV` online. Users can learn how to perform tasks such as object detection, segmentation, 3D morphological feature extraction, and visualization.
For users who wish to train `ShinyAnimalCV` with their own data, [the second section](#section-2) offers detailed instructions on training computer vision models using custom data. These guidelines walk users through the entire process, from data preparation to model training, ensuring that they can effectively train the models with their data.
[The third section](#section-3) focuses on incorporating the trained computer vision models into `ShinyAnimalCV` for local inference. By following the provided instructions, users will be able to seamlessly integrate their trained models into `ShinyAnimalCV`, enabling them to perform powerful and efficient inferences locally.

<!-- Section 1 -->
## Section 1: Video demo for online usage of ShinyAnimalCV<a name="section-1"></a>
We provide two video demos below to showcase the usage of two modules of [ShinyAnimalCV](https://shinyanimalcv.rc.ufl.edu/) online: object detection and segmentation, as well as 3D morphological feature extraction. Additionally, we offer [example images](https://github.com/uf-aiaos/ShinyAnimalCV/tree/main/ExampleData/Example_images_for_testing_ShinyAnimalCV) for users to explore these modules.

#### 1.1 Object detection and segmentation.<a name="subsection-1-1"></a> [[Video demo](https://youtu.be/FtR0YC9jCz0)]

#### 1.2 Three-dimensional morphological feature extraction and visualization.<a name="subsection-1-2"></a> [[Video demo](https://youtu.be/M6kmPiyYlPk)]

<!-- Section 2 -->

## Section 2: Training computer vision models with custom data <a name="section-2"></a>

### 2.1 Image annotation<a name="section-2-1"></a>
The first step in training computer vision models involves labeling or annotating objects in images. Here, we introduce [LabelMe](https://github.com/wkentaro/labelme), which is an open-source graphical image annotation tool that allows users to draw  polygons, rectangles, circles, lines and points to label the objects in images. When training the models using the data annotated from `LabelMe`, one of the inputs is an `info.yaml` file. However, it is important to note that the current version of `LabelMe` on GitHub does not directly return a `info.yaml` file. To overcome this, the following tutorials in this section will guide you through the process of downloading `LabelMe` from GitHub, making the necessary modifications to enable the generation of a `info.yaml` file, and building the modified standalone executable. For more details about building a standalone executable `LabelMe`, please refer to the file [Install-StandAlone-labelme.txt](https://github.com/uf-aiaos/ShinyAnimalCV/blob/main/Labelme/Install-StandAlone-labelme.txt) and [LabelMe GitHub Repo](https://github.com/wkentaro/labelme). 

#### 2.1.1 Create a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) with Python installed 
```bash
# create a conda env named `labelme`
conda create --name labelme python=3.9
# activate the created `labelme`
conda activate labelme
```

#### 2.1.2 Install two python packages 
```bash
conda install pyqt  
conda install -c conda-forge pyside2
```

#### 2.1.3 Download LabelMe from GibHub to local using [git](https://github.com/git-guides/install-git)
```bash
# download the LabelMe from GitHub to local
git clone https://github.com/wkentaro/labelme.git
# change directory to the downloaed labelme/ folder 
cd labelme/
```

#### 2.1.4 Modify the `json_to_dataset.py` file to enable the generation of a `info.yaml` file
```bash
# add this line at the beginning of `json_to_dataset.py`
import yaml 
```
```bash
# add the following codes before the line `logger.info("Saved to: {}".format(out_dir))` in `json_to_dataset.py`
logger.warning('info.yaml is being replaced by label_names.txt')
info = dict(label_names=label_names)
with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
	yaml.safe_dump(info, f, default_flow_style=False)
``` 

#### 2.1.5 Build the standalone executable LabelMe
```bash
pip install .                 
pip install pyinstaller
pyinstaller labelme.spec
``` 

####  2.1.6 Check the installed LabelMe version
```bash
labelme --version
```

#### 2.1.7 Label images using the standalone executable LabelMe 

<img src='https://github.com/uf-aiaos/ShinyAnimalCV/blob/main/ExampleData/Images_for_demonstration/twopigs_clean.gif?raw=true' width='100%' height='100%'>

More details about how to label images can be found on [LabelMe GitHub Repo](https://github.com/wkentaro/labelme).

#### 2.1.8 Covert the `.json` file returned from the previous step to dataset
```bash
labelme_json_to_dataset image_name.json -o image_name_json
```

The following five files will be found under the folder `image_name_json/`:
<img src='https://github.com/uf-aiaos/ShinyAnimalCV/blob/main/ExampleData/Images_for_demonstration/twopigs_json.png?raw=true' width='100%' height='100%'>

### 2.2 Model training and evaluation<a name="section-2-2"></a>

This section provides a step-by-step guide to training a computer vision model for object detection and instance segmentation using a modified state-of-the-art computer vision package, [Mask R-CNN](https://github.com/matterport/Mask_RCNN). Specifically, we have made modifications to the original `Mask R-CNN` to ensure compatibility with tensorflow (version 2.9.0 and 2.10.0), as the current GitHub version only supports tensorflow version <= 2.5.0. Additionally, we have replaced the original axis-aligned rectangle bounding box from `Mask R-CNN` with a minimum rotated rectangle bounding box using [OpenCV](https://opencv.org/). This enhancement allows for the extraction of more precise morphological features. The modified `Mask R-CNN` is available under [`mrcnn/`](https://github.com/uf-aiaos/ShinyAnimalCV/tree/main/ShinyAnimalCV_SourceCode/mrcnn). 

#### 2.2.1 Dataset preparation
This subsection details the process of preparing the training, validation, and testing sets for model training and evaluation. First, users need to partition the entire dataset into training, validation, and testing sets using a specified ratio, such as 7:2:1. Additionally, the `label.png` file within each `image_name_json` folder, obtained from `LabelMe`, needs to be transferred to a new mask folder named `cv2_mask` and renamed as `image_name.png`. To facilitate this procedure, we have included a script called [generate_cv2_mask.ipynb](https://github.com/uf-aiaos/ShinyAnimalCV/blob/main/MaskRCNN/generate_cv2_mask.ipynb), which can be used iteratively to prepare the `cv2_mask` folder. For more comprehensive details and explanations, please refer to the [generate_cv2_mask.ipynb](https://github.com/uf-aiaos/ShinyAnimalCV/blob/main/MaskRCNN/generate_cv2_mask.ipynb) file. Furthermore, we have provided a [demo dataset](https://github.com/uf-aiaos/ShinyAnimalCV/tree/main/ExampleData/Example_images_for_training_and_evaluating_model), which serves as a reference for dataset preparation.

#### 2.2.2 Train and evaluate Mask R-CNN model <a name="section-2-2-2"></a>
In this subsection, we show how to train and evaluate the model with organized annotated custom data using our provided Jypyter Notebook ([MaskRCNN.ipynb](https://github.com/uf-aiaos/ShinyAnimalCV/blob/main/MaskRCNN/MaskRCNN.ipynb)). 

- Create a conda environment named `maskrcnn` and install all dependencies for running the [MaskRCNN.ipynb](https://github.com/uf-aiaos/ShinyAnimalCV/blob/main/MaskRCNN/MaskRCNN.ipynb). The required dependencies file of `environment.yml` can be downloaded [here](https://github.com/uf-aiaos/ShinyAnimalCV/blob/main/MaskRCNN/environment.yml).
```bash
# create a conda environment `maskrcnn` and install all dependencies using `environment.yml`
conda env create -f environment.yml
```

- Open the [MaskRCNN.ipynb](https://github.com/uf-aiaos/ShinyAnimalCV/blob/main/MaskRCNN/MaskRCNN.ipynb) using [Jupyter Notebook](https://jupyter.org/) and follow the detailed instructions in `MaskRCNN.ipynb` to train and evaluate the model. 
```bash
# activate `maskrcnn` (if deactivated)
conda activate maskrcnn
# open `MaskRCNN.ipynb` 
jupyter notebook MaskRCNN.ipynb
```

## Section 3: Local inference with trained models in ShinyAnimalCV <a name="section-3"></a>
To enable users to perform local inference using the trained model on custom data while leveraging their local computational resources, this section provides detailed instructions on how to deploy and run ShinyAnimalCV locally with custom model weights. The source code for `ShinyAnimalCV` can be accessed at [ShinyAnimalCV_SourceCode](https://github.com/uf-aiaos/ShinyAnimalCV/blob/main/ShinyAnimalCV_SourceCode/). The nine pre-trained models included in online ShinyAnimalCV can be accessed on [Zenodo](https://zenodo.org/record/8196421).

### 3.1 Download `ShinyAnimalCV` in local 
```bash
# download this github repository in local using git
# the source code of ShinyAnimalCV is under ShinyAnimalCV/ShinyAnimalCV_SourceCode/
git clone https://github.com/uf-aiaos/ShinyAnimalCV.git
```

### 3.2 Set up running environment 
`ShinyAnimalCV` is built through the integration of R and Python using the R package [reticulate](https://rstudio.github.io/reticulate/). In the server.R file, we include the command `reticulate::use_condaenv(condaenv="maskrcnn", conda="auto", required=TRUE)` to ensure interoperability between R and Python. If you have not created the `maskrcnn` environment yet, please refer to [2.2.2 Train and evaluate Mask R-CNN model ](#section-2-2-2) for instructions on creating the maskrcnn Python environment.

### 3.3 Add custom trained weight into `ShinyAnimalCV`
To use the custom trained model weight for local inference using `ShinyAnimalCV`, users need to move custom weight file (.h5 file) obtained from Section 2 under `ShinyAnimalCV/ShinyAnimalCV_SourceCode/logs`and rename it as `CustomWeights.h5`. 

### 3.4 Deploy ShinyAnimalCV and perform inferences locally using the custom trained weight
The ShinyAnimalCV application can be deployed by running the following command within RStudio. Alternatively, users can open the `server.R` file with RStudio and click on the "Run App" button located at the top-right corner of the RStudio interface to deploy the application.
```bash
shiny::runApp('path_to_the_folder_ShinyAnimalCV_SourceCode')
```
Once the ShinyAnimalCV application is successfully deployed, a web-based graphical user interface will open, allowing users to interact with the application. In the interface, users can select their custom trained weights by choosing the option `CustomWeights` under "Step 2: Select a pretrained machine learning model." This enables users to perform inference using their custom trained weights.

<!-- contact and help -->
## Contact information and help <a name="section-contact"></a>

- Jin Wang ([jin.wang@ufl.edu](mailto:jin.wang@ufl.edu))
- Haipeng Yu ([haipengyu@ufl.edu](mailto:haipengyu@ufl.edu))

<!-- References -->
## References <a name="section-reference"></a>
If you use the materials provided in this repository, we kindly ask you to cite our paper: 

- Jin Wang, Yu Hu, Lirong Xiang, Gota Morota, Samantha A. Brooks, Carissa L. Wickens, Emily K. Miller-Cushon and Haipeng Yu. Technical note: ShinyAnimalCV: open-source cloud-based web application for object detection, segmentation, and three-dimensional visualization of animals using computer vision. <i>Journal of Animal Science.<i> [doi: 10.1093/jas/skad416](https://doi.org/10.1093/jas/skad416)

<!-- License -->
## License <a name="section-license"></a>
This project is primarily licensed under the GNU General Public License version 3 (GPLv3). 
