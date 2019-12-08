from sklearn.metrics import confusion_matrix, accuracy_score
import sys
import csv
from src.data.make_dataset import load_subjects, load_dataset

csvfile = sys.argv[1]

subjects = load_subjects('data/hcp-eval.csv')

with open(csvfile) as fd:
    reader = csv.DictReader(fd)
    ageref = []
    agepredicted = []
    for row in reader:
        id = row['id'].split('-')[0]
        ageref.append(subjects[id]['age'])
        agepredicted.append(row['age'])

score = accuracy_score(ageref, agepredicted)

print(score)