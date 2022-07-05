##### ___________________ Amsterdam
#dasy
python main.py --attr_value children students mobadults nmobadults elderly sur ant mar tur nonwestern western autoch --city ams --group_split 5 12 --nmodelpred 1 --popraster GHS_POP_100_near_cubicspline.tif --key Buurtcode --run_Pycno no --run_Dasy yes --run_Disaggregation no --iterMax 2 --methodopts aprf --ymethodopts Dasy --inputDataset AIL1 
#aprf
python main.py --attr_value children students mobadults nmobadults elderly sur ant mar tur nonwestern western autoch --city ams --group_split 5 12 --nmodelpred 1 --popraster GHS_POP_100_near_cubicspline.tif --key Buurtcode --run_Pycno no --run_Dasy no --run_Disaggregation yes --iterMax 2 --methodopts aprf --ymethodopts Dasy --inputDataset AIL1 
#apcnn
python main.py --attr_value children students mobadults nmobadults elderly sur ant mar tur nonwestern western autoch --city ams --group_split 5 12 --nmodelpred 1 --popraster GHS_POP_100_near_cubicspline.tif --key Buurtcode --run_Pycno no --run_Dasy no --run_Disaggregation yes --iterMax 2 --methodopts apcnn --ymethodopts Dasy --inputDataset AIL1 

##### ___________________ Copenhagen
python main.py --attr_value children students mobadults nmobadults elderly DNK EEU EurNonEU MENAP OthNonWest OthWestern TUR WEU --city cph --group_split 5 13 --nmodelpred 2 --popraster GHS_POP_100_near_cubicsplineWaterIESM_new.tif --key KOMNAME --run_Pycno no --run_Dasy no --run_Disaggregation yes --maxIters 10 --methodopts apcnn --ymethodopts Dasy --inputDataset AIL8   

##### ___________________ Krakow
python main.py --attr_value totalpop totalmig --city crc --group_split 5 12 --popraster GHS_POP_100_near_cubicsplineWaterIESM_new.tif --key DZIELNICA --run_Pycno no --run_Dasy no --run_Disaggregation no --maxIters 2 --methodopts apcnn --ymethodopts Dasy --inputDataset AIL12 


python main_eval.py --attr_value mar tur  --city ams --key Buurtcode --methodopts apcnn --verMassPreserv no --run_Evaluation yes --calc_Metrics yes --calc_Corr yes --plot_evalMaps yes --calc_Metrics_knn yes --plot_evalMaps_knn yes --plot_Matrices yes

##### ___________________ Evaluation ___________________ #####

##### ___________________ Amsterdam
# Run metrices
python main_eval.py --attr_value totalpop children students mobadults nmobadults elderly sur ant mar tur nonwestern western autoch --city ams --key Buurtcode --methodopts apcnn --verMassPreserv no --run_Evaluation yes --calc_Metrics yes --calc_Corr yes --plot_evalMaps no --calc_Metrics_knn no --plot_evalMaps_knn no --plot_Matrices no
# Plot maps
python main_eval.py --attr_value mar nonwestern --city ams --key Buurtcode --methodopts apcnn --verMassPreserv no --run_Evaluation yes --calc_Metrics no --calc_Corr no --plot_evalMaps yes --calc_Metrics_knn no --plot_evalMaps_knn yes --plot_Matrices no
#
python main_eval.py --attr_value mar nonwestern --city ams --key Buurtcode --methodopts apcnn --verMassPreserv no --run_Evaluation yes --calc_Metrics no --calc_Corr no --plot_Scatter yes --plot_evalMaps no --calc_Metrics_knn no --plot_evalMaps_knn no --plot_Matrices no
# Plot matrices
python main_eval.py --attr_value mar nonwestern --city ams --key Buurtcode --methodopts apcnn --verMassPreserv no --run_Evaluation yes --calc_Metrics no --calc_Corr no --plot_Scatter no --plot_evalMaps yes --calc_Metrics_knn no --plot_evalMaps_knn yes --plot_Matrices yes

python main_eval.py --attr_value mar nonwestern --city ams --key Buurtcode --methodopts apcnn --verMassPreserv no --run_Evaluation yes --calc_Metrics no --calc_Corr no --plot_Scatter no --plot_evalMaps no --calc_Metrics_knn no --plot_evalMaps_knn no --plot_Matrices yes


##### ___________________ Copenhagen
# Run metrices
python main_eval.py --attr_value children students mobadults nmobadults elderly DNK EEU EurNonEU MENAP OthNonWest OthWestern TUR WEU --city cph --key KOMNAME --methodopts apcnn --verMassPreserv no --run_Evaluation yes --calc_Metrics yes --calc_Corr yes --plot_evalMaps no --calc_Metrics_knn no --plot_evalMaps_knn no --plot_Matrices no

##### ___________________ Krakow