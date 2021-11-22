## Object recognition and computer vision 2021/2022

### Assignment 3: Image classification 

#### Create virtual environment and install requirements

```python 
python3 -m venv env
source env/bin/activate
```

```python
pip install torch # necessary to install a first version of it because of Detectron2
pip install -r requirements.txt
```

#### Dataset
We will be using a dataset containing 200 different classes of birds adapted from the [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
Download the training/validation/test images from [here](https://www.di.ens.fr/willow/teaching/recvis18orig/assignment3/bird_dataset.zip). The test image labels are not provided.

#### Small EDA on the data

Take a look at the notebook ``EDA.ipynb``.

#### Report

The overall method and results are described in the ```report.pdf```.

#### Run Mask R-CNN on all images and save newly cropped data

```python
python3 -m main_detector.py
```

#### Run feature extractor to get 2048-feature vector for each image

```python
python3 -m main_feature_extractor.py
```

Examples on how to use these embeddings can be found in the ``notebooks`` demo.

### Train the selected model on train set and evaluate it on val set

This script will train the defined model without previously computed embeddings.
```python
python3 -m main_classifier_without_embeddings.py
```

### Train the selected model using  Cross Validation

Using cross validation (CV) to train the selected model.

```python
python3 -m main_kfolds.py
```

#### Acknowledgments
Adapted from Rob Fergus and Soumith Chintala https://github.com/soumith/traffic-sign-detection-homework.<br/>
Adaptation done by Gul Varol: https://github.com/gulvarol
