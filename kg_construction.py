import numpy as np
import random
import math
import time
import pandas as pd


class Kg_construct_ehr():
    """
    construct knowledge graph out of EHR data
    """
    def __init__(self):
        file_path = '/datadrive/user_tingyi.wanyan/tensorflow_venv/registry_2020-06-29'
        self.reg = file_path + '/registry.csv'
        self.covid_lab = file_path + '/covid19LabTest.csv'
        self.lab = file_path + '/Lab.csv'
        self.vital = file_path + '/vitals.csv'
        file_path_ = '/home/tingyi.wanyan'
        self.lab_comb = file_path_ + '/lab_mapping_comb.csv'

    def read_csv(self):
        self.registry = pd.read_csv(self.reg)
        self.covid_labtest = pd.read_csv(self.covid_lab)
        self.labtest = pd.read_csv(self.lab)
        self.vital_sign = pd.read_csv(self.vital)
        self.lab_comb = pd.read_csv(self.lab_comb)
        self.reg_ar = np.array(self.registry)
        self.covid_ar = np.array(self.covid_labtest)
        self.labtest_ar = np.array(self.labtest)
        self.vital_sign_ar = np.array(self.vital_sign)
        self.lab_comb_ar = np.array(self.lab_comb)
    
    def create_kg_dic(self):
        self.dic_patient = {}
        self.dic_vital = {}
        self.dic_lab = {}
        self.dic_lab_category = {}
        self.crucial_vital = ['CAC - BLOOD PRESSURE','CAC - TEMPERATURE','CAC - PULSE OXIMETRY',
                'CAC - RESPIRATIONS','CAC - PULSE','CAC - HEIGHT','CAC - WEIGHT/SCALE']
        index_keep = np.where(self.lab_comb_ar[:,-1]==1)[0]
        self.lab_comb_keep = self.lab_comb_ar[index_keep]
        index_name = np.where(self.lab_comb_keep[:,-2]==self.lab_comb_keep[:,-2])[0]
        self.lab_test_feature = []
        [self.lab_test_feature.append(i) for i in self.lab_comb_keep[:,-2] if i not in self.lab_test_feature]
        self.lab_comb_keep_ = self.lab_comb_keep[index_name]
        self.cat_comb = self.lab_comb_keep[:,[0,-2]]
        for i in range(index_name.shape[0]):
            name_test = self.lab_comb_keep[i][0]
            name_category = self.lab_comb_keep[i][-2]
            if name_test not in self.dic_lab_category.keys():
                self.dic_lab_category[name_test] = name_category
                if name_category not in self.dic_lab:
                    self.dic_lab[name_category] = {}
                    #self.dic_lan[name_category]['specific name']={}
                    self.dic_lab[name_category].setdefault('specific_name',[]).append(name_test)
                else:
                    self.dic_lab[name_category].setdefault('specific_name',[]).append(name_test)
        """
        create initial vital sign dictionary
        """
        index_vital = 0
        for i in self.crucial_vital:
            if i == 'CAC - BLOOD PRESSURE':
                self.dic_vital['high']={}
                self.dic_vital['high']['index']=index_vital
                index_vital += 1
                self.dic_vital['low']={}
                self.dic_vital['low']['index']=index_vital
                index_vital += 1
            else:
                self.dic_vital[i]={}
                self.dic_vital[i]['index']=index_vital
                index_vital += 1


        

        icu = np.where(self.reg_ar[:,29]==self.reg_ar[:,29])[0]
        for i in icu:
            mrn_single = self.reg_ar[i,45]
            in_time_single = self.reg_ar[i,29]
            if self.reg_ar[i,11] == self.reg_ar[i,11]:
                death_flag = 1
            else:
                death_flag = 0
            #self.dic_patient[mrn_single]={}
            #self.dic_patient[mrn_single]['in_icu_time']=in_time_single
            #self.dic_patient[mrn_single]['death_flag']=death_flag
            self.in_time = in_time_single.split(' ')
            in_date = [np.int(i) for i in self.in_time[0].split('-')]
            in_date_value = (in_date[0]*365.0 + in_date[1]*30 + in_date[2])*24*60
            self.in_time_ = [np.int(i) for i in self.in_time[1].split(':')[0:-1]]
            in_time_value= self.in_time_[0]*60.0 + self.in_time_[1]
            total_in_time_value = in_date_value+in_time_value
            if mrn_single not in self.dic_patient.keys():
                self.dic_patient[mrn_single]={}
                self.dic_patient[mrn_single]['in_icu_time']=self.in_time
                self.dic_patient[mrn_single]['in_date']=in_date
                self.dic_patient[mrn_single]['in_time']=self.in_time_
                self.dic_patient[mrn_single]['death_flag']=death_flag
                self.dic_patient[mrn_single]['total_in_time_value']=total_in_time_value
                self.dic_patient[mrn_single]['prior_time_vital']={}
                self.dic_patient[mrn_single]['prior_time_lab']={}
                #self.dic_patient[mrn_single]['time_capture']={}
        mrn_icu = self.reg_ar[:,45][icu]
        covid_detect = np.where(self.covid_ar[:,7]!='NOT DETECTED')[0]
        covid_mrn = self.covid_ar[:,0][covid_detect]
        self.total_data = np.intersect1d(list(covid_mrn),list(mrn_icu))
        index = 0
        for i in self.dic_lab.keys():
            test_specific = self.lab_comb_keep_[np.where(self.lab_comb_keep_[:,-2]==i)[0]][:,0]
            num = 0
            test_patient_specific = []
            for j in test_specific:
                test_patient_specific += list(self.labtest_ar[np.where(kg.labtest_ar[:,2]==j)[0]][:,0])
            num += len(np.intersect1d(list(test_patient_specific),self.total_data))
            self.dic_lab[i]['num_patient']=num
            
        
        for i in self.total_data:
            in_icu_date = self.reg_ar
            self.single_patient_vital = np.where(self.vital_sign_ar[:,0]==i)[0]
            in_time_value = self.dic_patient[i]['total_in_time_value']
            self.single_patient_lab = np.where(self.labtest_ar[:,0]==i)[0]
           # print(index)
            #index += 1
            for j in self.single_patient_vital:
                obv_id = self.vital_sign_ar[j][2]
                if obv_id in self.crucial_vital:
                    self.check_data = self.vital_sign_ar[j][4]
                    self.dic_patient[i].setdefault('time_capture',[]).append(self.check_data)
                    date_year_value = float(str(self.vital_sign_ar[j][4])[0:4])*365
                    date_day_value = float(str(self.check_data)[4:6])*30+float(str(self.check_data)[6:8])
                    date_value = (date_year_value+date_day_value)*24*60
                    date_time_value = float(str(self.check_data)[8:10])*60+float(str(self.check_data)[10:12])
                    total_time_value = date_value+date_time_value
                    self.prior_time = np.int(np.floor(np.float((total_time_value-in_time_value)/60)))
                    if self.prior_time < 0:
                        continue
                    if obv_id == 'CAC - BLOOD PRESSURE':
                        self.check_obv = obv_id
                        self.check_ar = self.vital_sign_ar[j]
                        self.check_value_presure = self.vital_sign_ar[j][3]
                        try:
                            value = self.vital_sign_ar[j][3].split('/')
                        except:
                            continue
                        if self.check_value_presure == '""':
                            continue
                        if self.prior_time not in self.dic_patient[i]['prior_time_vital']:
                            self.dic_patient[i]['prior_time_vital'][self.prior_time]={}
                            self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault('high',[]).append(value[0])
                            self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault('low',[]).append(value[1])
                        else:
                            self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault('high',[]).append(value[0])
                            self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault('low',[]).append(value[1])
                        self.dic_vital['high'].setdefault('value',[]).append(value[0])
                        self.dic_vital['low'].setdefault('value',[]).append(value[1])
                    else:
                        self.check_value = self.vital_sign_ar[j][3]
                        self.check_obv = obv_id
                        self.check_ar = self.vital_sign_ar[j]
                        if self.check_value == '""':
                            continue
                        value = float(self.vital_sign_ar[j][3])
                        if np.isnan(value):
                            continue
                        if self.prior_time not in self.dic_patient[i]['prior_time_vital']:
                            self.dic_patient[i]['prior_time_vital'][self.prior_time]={}
                            self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault(obv_id,[]).append(value)
                        else:
                            self.dic_patient[i]['prior_time_vital'][self.prior_time].setdefault(obv_id,[]).append(value)
                        self.dic_vital[obv_id].setdefault('value',[]).append(value)
                    
        """
        res = []        
        [res.append(x) for x in list(self.labtest_ar[:,2]) if x not in res]
        res_cur = []
        [res_cur.append(x) for x in res if x==x]
        for i in range(len(res_cur)):
            num = np.intersect1d(list(kg.labtest_ar[np.where(kg.labtest_ar[:,2]==res_cur[i])[0]][:,0]),list(kg.total_data)).shape[0]
            res_cur[i] = res_cur[i]+' ' + str(num)
        """
            

if __name__ == "__main__":
    kg = Kg_construct_ehr()
    kg.read_csv()
    kg.create_kg_dic()
