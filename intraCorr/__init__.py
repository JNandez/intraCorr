import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import f as fdist

import matplotlib.pyplot as plt
import matplotlib.cm as cm

pd.options.display.float_format = '{:,.4f}'.format

class intraCorr():
    """
    This class calculates the intra-individual level or logitudinal change. It makes use of the pandas module.
    Inputs:
       * subject (str): name of dataframe column with the individuals/subjects
       * independent (str): name of the dataframe column that will be treated as independent variable 
       * dependent (str): name of the dataframe column that will be treated as dependent variable
       * data (pandas.core.frame.DataFrame): pandas dataframe with the data
       * njobs (int): numbers of CPU cores to use for fitting the multiple regression
       * prefix (str): prefix for the subject in case they are numbers. 
       
    """
    def __init__(self,subject, independent, dependent, data, njobs = 1, prefix = 'Subject'):
        self.njobs = njobs
        self.subject = subject
        self.independent = independent
        self.dependent = dependent
        self.data = data
        self.uniqueSubjects = self.data[self.subject].unique()
        self.prefix = prefix
        self.trainData = data.join(pd.get_dummies(data[self.subject], prefix = self.prefix))
        self.X_t = self.trainData[['{0}_{1}'.format(self.prefix,i) for i in self.uniqueSubjects]+[self.independent]]
        self.X_m = self.trainData[self.independent].values.reshape(-1,1)
        self.X_s = self.trainData[['{0}_{1}'.format(self.prefix,i) for i in self.uniqueSubjects]]
        self.y   = self.trainData[self.dependent]
        self.df_t = self.X_t.shape[0]-len(self.X_t.columns)
        self.df_m = len(self.X_s.columns)-1
        self.df_s = 1
    def fit(self):
        # use independet variable + subjects 
        self.reg_t = LinearRegression(n_jobs = self.njobs).fit(self.X_t, self.y)
        self.preds_t = self.reg_t.predict(self.X_t)
        self.RSS_t = ((self.preds_t-self.y)**2).sum()
        self.signCorr = np.sign(self.reg_t.coef_[-1])
        #only independent variable
        self.reg_m = LinearRegression(n_jobs = self.njobs).fit(self.X_m, self.y)
        self.preds_m = self.reg_m.predict(self.X_m)
        self.RSS_m = ((self.preds_m - self.y)**2).sum()
        self.SS_m = ((self.preds_m - self.preds_t)**2).sum()
        #only subjects
        self.reg_s = LinearRegression(n_jobs = self.njobs).fit(self.X_s, self.y)
        self.preds_s = self.reg_s.predict(self.X_s)
        self.RSS_s = ((self.preds_s - self.y)**2).sum()
        self.SS_s = ((self.preds_s - self.preds_t)**2).sum()
        ## MSE
        self.MSE_s = self.SS_s/self.df_s
        self.MSE_m = self.SS_m/self.df_m
        self.MSE_t = self.RSS_t/self.df_t
        ## F ratio
        self.F_ratio_m = self.MSE_m/self.MSE_t
        self.F_ratio_s = self.MSE_s/self.MSE_t
        ## p value
        self.p_value_m = fdist.sf(self.F_ratio_m,self.df_m,self.df_t)
        self.p_value_s = fdist.sf(self.F_ratio_s,self.df_s,self.df_t)
        ##final DF
        self.df_total = self.df_m+self.df_s+self.df_t
        self.ss_total = ((self.y-self.y.mean())**2).sum()
        self.mse_total = self.ss_total/self.df_total
        self.result = pd.DataFrame({'DoF':[self.df_m,self.df_s,self.df_t,self.df_total],
                                    'SumOfSq':[self.SS_m,self.SS_s,self.RSS_t,self.ss_total],
                                    'MSE':[self.MSE_m,self.MSE_s,self.MSE_t,self.mse_total],
                                    'F_value':[self.F_ratio_m,self.F_ratio_s,None,None],
                                    'p_value':['<0.0001' if self.p_value_m < 0.0001 else self.p_value_m,self.p_value_s,None,None]
                                   },
                                   index = [self.subject,self.independent,'Residual','Total']
                                   )
        self.corr = self.signCorr*np.sqrt(self.SS_s/(self.SS_s + self.RSS_t))
        return self.corr, self.p_value_s
    def get_model(self):
        self.fit()
        return self.reg_t,self.trainData

    def plot(self,figsize=(10,10),prefix = None):
        self.fit()
        colors = cm.coolwarm(np.linspace(0, 1, len(self.uniqueSubjects)))
        fig, ax = plt.subplots(figsize=figsize)
        for k,sub_k in enumerate(self.uniqueSubjects):
            data_tmp = self.trainData.loc[self.trainData[self.subject] == sub_k, ['{0}_{1}'.format(self.prefix,i) for i in self.uniqueSubjects]+[self.independent]]
            y_tmp = self.trainData.loc[self.trainData[self.subject] == sub_k,self.dependent]
            preds_tmp = self.reg_t.predict(data_tmp)
            ax.plot(data_tmp[self.independent],preds_tmp,color = colors[k])
            if prefix is not None:
                label_k = prefix + '_' + str(sub_k)
            else:
                label_k = str(sub_k)
            ax.scatter(data_tmp[self.independent],y_tmp, color = colors[k], label = label_k)
        plt.xlabel(self.independent)
        plt.ylabel(self.dependent)
        plt.title('Correlation = {0:.3f}, P Value = {1:.3f}'.format(self.corr, self.p_value_s))
        plt.legend(loc='best',fontsize='small')

    def print_result(self):
        self.fit()
        print(self.result)

    def bland_altman_1995():
        data_raw = [{'Subject': 1, 'pH': 6.68, 'PaCO2': 3.97},
                    {'Subject': 1, 'pH': 6.53, 'PaCO2': 4.12},
                    {'Subject': 1, 'pH': 6.43, 'PaCO2': 4.09},
                    {'Subject': 1, 'pH': 6.33, 'PaCO2': 3.97},
                    {'Subject': 2, 'pH': 6.85, 'PaCO2': 5.27},
                    {'Subject': 2, 'pH': 7.06, 'PaCO2': 5.37},
                    {'Subject': 2, 'pH': 7.13, 'PaCO2': 5.41},
                    {'Subject': 2, 'pH': 7.17, 'PaCO2': 5.44},
                    {'Subject': 3, 'pH': 7.4, 'PaCO2': 5.67},
                    {'Subject': 3, 'pH': 7.42, 'PaCO2': 3.64},
                    {'Subject': 3, 'pH': 7.41, 'PaCO2': 4.32},
                    {'Subject': 3, 'pH': 7.37, 'PaCO2': 4.73},
                    {'Subject': 3, 'pH': 7.34, 'PaCO2': 4.96},
                    {'Subject': 3, 'pH': 7.35, 'PaCO2': 5.04},
                    {'Subject': 3, 'pH': 7.28, 'PaCO2': 5.22},
                    {'Subject': 3, 'pH': 7.3, 'PaCO2': 4.82},
                    {'Subject': 3, 'pH': 7.34, 'PaCO2': 5.07},
                    {'Subject': 4, 'pH': 7.36, 'PaCO2': 5.67},
                    {'Subject': 4, 'pH': 7.33, 'PaCO2': 5.1},
                    {'Subject': 4, 'pH': 7.29, 'PaCO2': 5.53},
                    {'Subject': 4, 'pH': 7.3, 'PaCO2': 4.75},
                    {'Subject': 4, 'pH': 7.35, 'PaCO2': 5.51},
                    {'Subject': 5, 'pH': 7.35, 'PaCO2': 4.28},
                    {'Subject': 5, 'pH': 7.3, 'PaCO2': 4.44},
                    {'Subject': 5, 'pH': 7.3, 'PaCO2': 4.32},
                    {'Subject': 5, 'pH': 7.37, 'PaCO2': 3.23},
                    {'Subject': 5, 'pH': 7.27, 'PaCO2': 4.46},
                    {'Subject': 5, 'pH': 7.28, 'PaCO2': 4.72},
                    {'Subject': 5, 'pH': 7.32, 'PaCO2': 4.75},
                    {'Subject': 5, 'pH': 7.32, 'PaCO2': 4.99},
                    {'Subject': 6, 'pH': 7.38, 'PaCO2': 4.78},
                    {'Subject': 6, 'pH': 7.3, 'PaCO2': 4.73},
                    {'Subject': 6, 'pH': 7.29, 'PaCO2': 5.12},
                    {'Subject': 6, 'pH': 7.33, 'PaCO2': 4.93},
                    {'Subject': 6, 'pH': 7.31, 'PaCO2': 5.03},
                    {'Subject': 6, 'pH': 7.33, 'PaCO2': 4.93},
                    {'Subject': 7, 'pH': 6.86, 'PaCO2': 6.85},
                    {'Subject': 7, 'pH': 6.94, 'PaCO2': 6.44},
                    {'Subject': 7, 'pH': 6.92, 'PaCO2': 6.52},
                    {'Subject': 8, 'pH': 7.19, 'PaCO2': 5.28},
                    {'Subject': 8, 'pH': 7.29, 'PaCO2': 4.56},
                    {'Subject': 8, 'pH': 7.21, 'PaCO2': 4.34},
                    {'Subject': 8, 'pH': 7.25, 'PaCO2': 4.32},
                    {'Subject': 8, 'pH': 7.2, 'PaCO2': 4.41},
                    {'Subject': 8, 'pH': 7.19, 'PaCO2': 3.69},
                    {'Subject': 8, 'pH': 6.77, 'PaCO2': 6.09},
                    {'Subject': 8, 'pH': 6.82, 'PaCO2': 5.58}]
        return pd.DataFrame(data_raw)
