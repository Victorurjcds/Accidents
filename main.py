"""
Created on Thu Mar 23 21:47:23 2017

@author: Victor Suárez Gutiérrez
Research Assistant. Data Scientist at URJC/HGUGM.
Contact: ssuarezvictor@gmail.com
"""

###############################################################################
###############################################################################
###############################################################################
###############################################################################

# Inicialization:
#%reset -f


# Define working directory
import os
os.chdir('C:/Users/Victor/Documents/Ambito_profesional/proyectos/accidentes')


# import libraries
from loading import load_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold as kfold
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import roc_auc_score as ras
import pickle 



###############################################################################
###############################################################################
# Cargar datos:
Acc_test = pd.Series(np.zeros((100)))
AUC = pd.Series(np.zeros((100)))

for j in range(100):
    np.random.seed(j+31)
    [data_original, data, output, features] = load_dataset()      # IMD: Intensidad Media Diaria de tráfico.                         
    X_train, X_test, y_train, y_test = train_test_split(data, output, test_size=0.3, random_state=31)
    X_train = X_train.reset_index()
    X_train = X_train.drop('index',axis=1)
    y_train = y_train.reset_index()
    y_train = y_train.drop('index',axis=1)

    # Normalization of IMD and GPS:
    average_x = X_train['GPS_x'].mean()
    stderror_x = X_train['GPS_x'].std()
    average_y = X_train['GPS_y'].mean()
    stderror_y = X_train['GPS_y'].std()
    average_z = X_train['GPS_z'].mean()
    stderror_z = X_train['GPS_z'].std()
    average_IMD = X_train['IMD'].mean()
    stderror_IMD = X_train['IMD'].std()
    X_train_norm = X_train.copy()
    X_test_norm = X_test.copy()
    X_train_norm['GPS_x'] = (X_train['GPS_x']-average_x)/stderror_z                              # only X_train dataset
    X_train_norm['GPS_y'] = (X_train['GPS_y']-average_y)/stderror_y
    X_train_norm['GPS_z'] = (X_train['GPS_z']-average_z)/stderror_z
    X_train_norm['IMD'] = (X_train['IMD']-average_IMD)/stderror_IMD
    
    X_test_norm['GPS_x'] = (X_test['GPS_x']-average_x)/stderror_z                              # only X_train dataset
    X_test_norm['GPS_y'] = (X_test['GPS_y']-average_y)/stderror_y
    X_test_norm['GPS_z'] = (X_test['GPS_z']-average_z)/stderror_z
    X_test_norm['IMD'] = (X_test['IMD']-average_IMD)/stderror_IMD


###############################################################################
###############################################################################
# Feature selection:    
    clf = rfc(n_estimators=100, max_features='auto', class_weight='balanced_subsample', n_jobs=-1, random_state=31)       # each tree is balanced.
    kf = kfold(n_splits=10, random_state=31)
    Acc = pd.Series(np.zeros((kf.n_splits)))
    feat_imp = pd.DataFrame(np.zeros((len(X_train_norm.ix[1,:]),kf.n_splits)))
    i=0
    for train_index, validation_index in kf.split(X_train_norm):
        X_tra, X_val = X_train_norm.ix[train_index,:], X_train_norm.ix[validation_index,:]
        y_tra, y_val = y_train.ix[train_index], y_train.ix[validation_index]  
        clf = clf.fit(X_tra, y_tra['Gravedad'])
        prediction = clf.predict(X_val)
        Acc.ix[i] = np.mean(np.array(y_val).T==prediction)
        feat_imp.ix[:,i] = clf.feature_importances_
        i +=1
    Acc_final_rf = np.mean(Acc)
    feat_imp = np.mean(feat_imp,axis=1)
    lista1, ordered_feat = zip(*sorted(zip(feat_imp, features), reverse=True))
    final = clf.predict(X_test_norm)
    Acc_test[j] = np.mean(final==y_test)
    AUC[j] = ras(y_test, final, average='macro')
    del clf,kf,Acc


# Final results:
auc = AUC.copy()
Acc_test.mean()
Acc_test.std()

plt.figure()
plt.plot(range(1,101), Acc_test)
plt.plot(range(1,101), np.tile(Acc_test.mean(),len(Acc_test)), c='r')
plt.plot(range(1,101), np.tile(Acc_test.mean()+Acc_test.std(),len(Acc_test)), c='r', ls='--')
plt.plot(range(1,101), np.tile(Acc_test.mean()-Acc_test.std(),len(Acc_test)), c='r', ls='--')
plt.title('Accuracy test')
plt.legend(['Acc','Mean','Std'])
plt.ylim(0.5,1)
plt.xlim(1,100)

plt.figure()
plt.plot(range(1,101), Acc_test)
plt.plot(range(1,101), np.tile(Acc_test.mean(),len(Acc_test)), c='r')
plt.plot(range(1,101), np.tile(Acc_test.mean()+Acc_test.std(),len(Acc_test)), c='r', ls='--')
plt.plot(range(1,101), np.tile(Acc_test.mean()-Acc_test.std(),len(Acc_test)), c='r', ls='--')
plt.title('Accuracy test')
plt.legend(['Acc','Mean','Std'])
plt.ylim(0.7,0.77)
plt.xlim(1,100)



###############################################################################
###############################################################################
pickle.dump([auc, Acc_test], open('results.p', 'wb'))
auc, Acc_test = pickle.load( open('results.p', 'rb' ) )
 


###############################################################################
###############################################################################
###############################################################################
###############################################################################


