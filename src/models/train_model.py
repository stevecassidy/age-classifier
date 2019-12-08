from src.data.preprocess import run_pca
from src.data.make_dataset import load_dataset, load_subjects
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

def transform_data(data, channels):
    """Select a number of channels from each data record and concatenate them into a single vector"""
    
    return [np.ravel(d[0][:channels]) for d in data['data']] 


dataset = load_dataset('data/processed/hcp-new', 'data/hcp-train.csv')
testdataset = load_dataset('data/processed/hcp-new', 'data/hcp-eval.csv')

channels = 10

dd = transform_data(dataset, channels)
td = transform_data(testdataset, channels)

model = svm.SVC(gamma=0.01, C=10.)
model.fit(dd, dataset['age'])

with open('models/svm-age.dat', 'wb') as out:
    pickle.dump(model, out)

predicted = model.predict(td)

print("id,age")
for pair in zip(testdataset['id'], predicted):
    print("{},{}".format(pair[0], pair[1]))




