# -*- coding: utf-8 -*-
import click
import logging
import csv
import numpy as np
import os
from typing import Dict
import mne
import glob

from src.features.build_features import spectral_epochs, read_hcp, read_mous, read_camcan

PROJECT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

HCP_DATA_FOLDER = PROJECT_DIR + "/data/raw/"
MOUS_DATA_FOLDER = PROJECT_DIR + "/data/Donders_MEG/"

HCP_DATA_FOLDER = '/var/syndisk/MEG/HCP/data/'
MOUS_DATA_FOLDER = '/var/syndisk/MEG/Donders_MEG/'
CAMCAN_DATA_FOLDER = "/var/syndisk/MEG/CANCAM/camcan43/cc700/meg/pipeline/release004/data/aamod_meg_get_fif_00001/"


EPOCH_DURATION = 60  # duration of recording (s) for each processed data point 
                     # recordings are cut up into chunks of this size 

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.ERROR, format=log_fmt)
mne.set_log_level('ERROR')
logger = logging.getLogger()


def load_dataset(data_folder: str, csvfile: str) -> Dict:
    """Load a dataset given a csv file containing subject ids and metadata
    Return a dictionary with keys 'target' and 'data' suitable for training a model"""

    result = {
        'id': [],
        'age': [],
        'gender': [],
        'data': [],
    }
    subjects: Dict = load_subjects(csvfile)
    
    for subject in subjects:
        # get all files matching this speaker
        pattern = os.path.join(data_folder, subject+"*")
        for filename in glob.glob(pattern):
            data = np.load(filename)
            result['id'].append(os.path.splitext(os.path.basename(filename))[0])
            if 'age' in subjects[subject]:
                result['age'].append(subjects[subject]['age'])  
                result['gender'].append(subjects[subject]['gender'])  
            result['data'].append(data)
    
    return result


def load_subjects(csvfile: str) -> Dict:
    """Load a list of subjects from a csv file along with metadata
    
    Subject,Age,Gender,Acquisition,Release
    195041,31-35,F,Q07,S500
    ...

    Return a dictionary with Subjects as keys and Age as the value
    """

    result: Dict = {}
    with open(csvfile, 'r', encoding='utf-8-sig') as fd:
        reader: csv.DictReader = csv.DictReader(fd)
        for row in reader:
            if 'Age' in row:
                result[row['Subject']] = {'age': row['Age'], 'gender': row['Gender']}
            else:
                result[row['Subject']] = {}

    return result

def make_hcp_dataset(csvfile: str, output_filepath: str) -> None:
    """Process the hcp dataset"""

    subjects = load_subjects(csvfile)

    for subject in subjects:
        print(os.path.join(HCP_DATA_FOLDER, subject))
        if os.path.exists(os.path.join(HCP_DATA_FOLDER, subject)):
            for run_index in range(3):
                raw = read_hcp(subject, HCP_DATA_FOLDER, run_index)
                labels, features = spectral_epochs(subject, raw, EPOCH_DURATION, max_freq=74)  # to match MOUS
                for i in range(len(labels)):
                    fname = os.path.join(output_filepath, labels[i] + "-" + str(run_index) + ".npy")
                    np.save(fname, features[i])
                    logger.info("Wrote {}".format(fname))
        else:
            logger.error("Missing: {}".format(subject))


def make_mous_dataset(csvfile: str, output_filepath: str) -> None:
    """Process the MOUS dataset"""

    subjects = load_subjects(csvfile)

    for subject in subjects:
        print("Subject", subject)
        raw = read_mous(subject, MOUS_DATA_FOLDER)
        print("read raw...")
        if raw:
            labels, features = spectral_epochs(subject, raw, EPOCH_DURATION, max_freq=74)
            for i in range(len(labels)):
                fname = os.path.join(output_filepath, labels[i] + ".npy")
                np.save(fname, features[i])
                print("Wrote {}".format(fname))
                logger.info("Wrote {}".format(fname))
        else:
            logger.error("Missing: {}".format(subject))



def make_camcan_dataset(csvfile: str, output_filepath: str) -> None:
    """Process the CAMCAN dataset"""

    subjects = load_subjects(csvfile)

    for subject in subjects:
        print("Subject", subject)
        raw = read_camcan(subject, CAMCAN_DATA_FOLDER)
        print("read raw...")
        if raw:
            labels, features = spectral_epochs(subject, raw, EPOCH_DURATION, filter=False)
            for i in range(len(labels)):
                fname = os.path.join(output_filepath, labels[i] + ".npy")
                np.save(fname, features[i])
                print("Wrote {}".format(fname))
                logger.info("Wrote {}".format(fname))
        else:
            logger.error("Missing: {}".format(subject))



@click.command()
@click.argument('dataset', type=click.STRING)
@click.argument('csvfile', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(dataset, csvfile, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)

    if dataset == 'hcp':
        make_hcp_dataset(csvfile, output_filepath)
    elif dataset == 'mous':
        make_mous_dataset(csvfile, output_filepath)
    elif dataset == 'camcan':
        make_camcan_dataset(csvfile, output_filepath)
    else:
        print("Unknown dataset name", dataset)

    print("\nDone")

if __name__ == '__main__':

    main()
