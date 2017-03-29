# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 21:05:20 2017

@author: Victor
"""

# Load data:
def load_dataset():   
    import scipy.io
    import numpy as np
    import pandas as pd
    
    mat = scipy.io.loadmat('Expedientes.mat')
    mat_expe = mat['Expedientes']
    count = 0
    names = pd.Series(np.zeros((len(mat_expe[0,0]))))
    data = pd.DataFrame(np.zeros((len(mat_expe), len(mat_expe[0,0]))))
    for i in range(len(data.ix[0,:])):
        names[i] = mat['Expedientes'].dtype.descr[i][0]
        try:
            data.ix[:,i] = list(map(lambda x: int(x), mat['Expedientes'][mat['Expedientes'].dtype.descr[i][0]]))
        except:
            try:
                data.ix[:,i] = list(map(lambda x: x[0][0], mat['Expedientes'][mat['Expedientes'].dtype.descr[i][0]]))
                data.ix[:,i] = np.nan
                count +=1
            except:
                data.ix[:,i] = mat['Expedientes'][mat['Expedientes'].dtype.descr[i][0]]

    data.columns = names
    data = data.dropna(axis=1)
    data_org = data.copy()
    data = data.drop(['Id','Numero_Accidente','Hora','Grado_Gravedad','Tipo_Est','N_Victimas','N_Graves','N_Muertos','N_Leves'], 1)
    unos = np.array(data[data['Gravedad']==1].index)
    zeros = pd.Series(data[data['Gravedad']==0].index, index=range(len(data[data['Gravedad']==0].index)))
    zeros_random = np.random.randint(0, high=len(zeros)-1, size=len(unos), dtype='int')  
    data = data.ix[np.concatenate((unos,zeros_random),axis=0),:]
    output = data['Gravedad']
    data = data.drop('Gravedad',1)
    features = list(data.columns)
    return (data_org, data, output, features)






       
        













