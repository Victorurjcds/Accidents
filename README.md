# Accidents
Estimation of car accidents importance. Database was supported by DGT.

We have several features about accidents such as: year, day, month, road surface, GPS coordinates, traffic index, etc. We have to estimate wether an accident is relevant or not.

Dataset: Expedientes.mat  
- Rows: 9341 samples
- Columns: 86 features


Note: N_Muertos, N_Graves, N_Leves, N_Victimas and Grado_Gravedad (importance level) should be deleted for data analysis because they are correlated with output (Gravedad features).



Results: 
- results.p: two pandas Series [Acc test and AUC] for 100 different subsamples.
- Acc: test accuracy
- Acc_zoom: test accuracy zoom
