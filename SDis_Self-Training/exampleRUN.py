python main.py -l children students mobadults nmobadults elderly sur ant mar tur nonwestern western autoch --city=ams --popraster=GHS_POP_100_near_cubicspline.tif --key=Buurtcode --run_Pycno=no --run_Dasy=yes --run_Disaggregation=no --iterMax=2 --methodopts=['aprf'] --inputDataset=['AIL1'] --verMassPreserv=no --run_Evaluation=no


python main.py -l children students mobadults nmobadults elderly sur ant mar tur nonwestern western autoch --city=ams --popraster=GHS_POP_100_near_cubicspline.tif --key=Buurtcode --run_Pycno=no --run_Dasy=no --run_Disaggregation=yes --iterMax=2 --methodopts=['aprf'] --inputDataset=['AIL1'] --verMassPreserv=no --run_Evaluation=no
#-------- GLOBAL ARGUMENTS --------
city ='ams'
popraster = 'GHS_POP_100_near_cubicspline.tif'.format(city) 
key = 'Buurtcode' #'BU_CODE'
  

#-------- SELECT DEMOGRAPHIC GROUP OR LIST OF GROUPS --------
# If it is a list it will be calculated in multi-output model 
# Total Population : 'totalpop',
# 5 Age Groups : 'children', 'students','mobadults', 'nmobadults', 'elderly', 
# 7 Migrant Groups : 'sur', 'ant', 'mar','tur', 'nonwestern', 'western', 'autoch'

#'totalpop', 'students','mobadults', 'nmobadults', 'elderly', 'sur', 'ant', 'mar','tur', 'nonwestern', 'western', 'autoch'
attr_value = ['children', 'students','mobadults', 'nmobadults', 'elderly', 'sur', 'ant', 'mar','tur', 'nonwestern', 'western', 'autoch' ]   
  
#-------- SELECT PROCESS --------
#1. CALCULATE SIMPLE HEURISTIC ESTIMATES WITH PYCHNOPHYLACTIC OR DASYMETRIC MAPPING
run_Pycno = 'no'
run_Dasy = 'yes'

#2. TRAIN REGRESSION MODEL (FURTHER CHOICES NEED TO BE DEFINED BELOW)
run_Disaggregation = 'no'
    # 2.1. SELECT METHOD/MODEL 
        # aplm (linear model) 
        # aprf (random forest) 
        # apcatbr (Catboost Regressor)
    # 2.2. SELECT DISAGGREGATED METHOD TO BE USED AS INPUT
        # Pycno
        # Dasy
    # 2.3 SELECT ANCILLARY DATASET

# 3. VERIFY MASS PRESERVATION
verMassPreserv = 'no'

# 4. EVALUATE RESULTS
run_EvaluationGC_ams = 'no'