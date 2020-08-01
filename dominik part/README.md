# cil2020-road-segmentation

Team Member: Xiao'ao Song, Pavel Pozdnyakov, Dominik Alberto

This repository contains the tools and models for the the course project of 
[Computational Intelligence Lab](http://da.inf.ethz.ch/teaching/2020/CIL/) (Spring 2020): Road Segmentaion.


# Dependencies

Cuda must be installed to be able to run the torch code.

* PyMaxflow (needs Tkinter)
* torch (Nightly version)
* torchvision (Nightly version)
* scipy
* numpy
* Pillow
* matplotlib
* tqdm

# Basemodels 

In our project we use the two baseline models introduced in the excersice 9.

The code can be found at https://github.com/dalab/lecture_cil_public/tree/master/exercises/2020/ex9

# How to run

To run the code with all default settings, place the folders test_images and training in the directory "ROOT_DIR/data".

The neural net can be trained and tested by executing the command below from the ROOT_DIR:

```
 python nn/main.py
```
Check "nn/main.py" to see all parameters that can be passed via command line.

To create an ensemble prediction with majority voting execute:

```
 python ensemble/main.py -i output_deeplab50 -i output_deeplab101 -i output_resnet50 -i output_resnet101
```
You can pass any number of output directory with "-i". To see further parameters (e.g. voting scheme) check "ensemble/main.py".

To postprocess the output with the graphcut algorithm run:

```
 python gc/main.py --verbose False
```

If you want to modify the default parameters, check the "main.py" files to see which parameters can be passed via the command line.

# See results

All outputs will be shown in the ROOT_DIR by default.
