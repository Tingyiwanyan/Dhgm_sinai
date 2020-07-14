import os
import json
import numpy as np
import random

class kg_process_data():
    """
    divide into train and test data set
    """
    def __init__(self,kg):
        self.train_percent = 0.7
        self.test_percent = 0.3
        self.kg = kg
        self.train_patient = []
        self.test_patient = []

    def separate_train_test(self):
        self.data_patient_num = len(self.kg.dic_patient.keys())
        self.train_num = np.int(np.floor(self.data_patient_num*self.train_percent))
        for i in self.kg.dic_patient.keys()[0:self.train_num]:
            self.train_patient.append(i)
        test_whole = [i for i in self.kg.dic_patient.keys() if i not in self.train_patient]
        for i in test_whole:
            self.test_patient.append(i)
