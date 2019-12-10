# Age and Gender Classification from MEG

Code to implement the baseline model for the MEG age and gender prediction task.  

## Installation

The code here requires MNE Python and a few other dependencies. It is best to install this into
a CONDA environment.  If you have gnu-make installed (eg. on a Mac or Linux) you can use the 
provided Makefile with the command:

```
    make environment
```

This will create a new conda environment called `age-classifier` and install the required packages.
To achieve the same thing without the makefile run the following commands:

```
# install MNE and requirements https://mne.tools/stable/install/mne_python.html
curl --remote-name https://raw.githubusercontent.com/mne-tools/mne-python/master/environment.yml
conda create --name age-classifier --file environment.yml
conda activate age-classifier
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

## Data

There is no data in this repository, you need to copy that from the sources provided in the Hackathon. 
The pre-processing scripts need to be told where the raw data is stored. The pre-processed data
can be copied into a folder `data` in this project - `data/preprocessed/hcp` and `data/preprocessed/mous`
contain the two datasets.  The CSV files listing subjects should be stored in the `data` folder. 

## Scripts

The `src` folder contains a number of python files implementing pre-processing scripts and a simple model. 

### `src/data/make_dataset.py`

This script converts the raw data into the preprocessed version. You need to edit the file to set the
location of the data on your system. You can then run it from the command line, eg:

```
python -m src.data.make_dataset hcp data/hcp-speakers.csv data/processed/hcp
```

The first argument is `hcp` or `mous` for the different datasets.  This script makes use of
`src/features/build_features.py` to compute the feature vector for each segment. If you want to
modify the feature vector, then make changes here.


### `src/models/train_model.py`

This script trains a simple SVM model on the HCP data and generates predictions on the evaluation data.

```
python -m src.models.train_model > hcp-prediction-results.csv
```

