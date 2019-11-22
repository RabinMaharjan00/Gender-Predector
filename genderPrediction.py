# -*- coding: utf-8 -*-
import random
import nltk
from nltk.corpus import names


def gender_features(name):
    return {'last_letter':name[-1]}

labeled_name = ([(name, "male") for name in names.words('male.txt')] + [(name,"female") for name in names.words('female.txt')])

random.shuffle(labeled_name)

featureset = [(gender_features(n),gender) for (n, gender) in labeled_name]

train_set, test_set = featureset[500:], featureset[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
data = input("Enter your name:")
output = classifier.classify(gender_features(data))
print(output)



