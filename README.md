# Spatial Disaggregation
(---Work in progress---)
This repository contains the code for the spatial disaggregation of population data from various administrative levels 100m. grid cells.
The followed methods are : single- and multi- output Random Forests and Gradient Boosting with Catboost.

You need to have the following:
1. An environment with the packages included in env.yml.
2. An AncillaryData folder with the desired ancillary data for each case study. GHS, CORINE LAND COVER, ESM etc are examples of ancillary data. 
You need to define them in runDisaggregation.py (raster format).
3. A SDis_Self-Training/Shapefiles/ with the vector layer of the administrative units (shp).
4. A SDis_Self-Training/Statistics/ folder with the aggregated counts respectively (csv).

The main architecture was designed by João Monteiro.
Further information can be found in '*Spatial Disaggregation of Population Subgroups Leveraging Self-Trained Multi-Output Gradient Boosted Regression Trees*'

*Acknowledgements*
This work has been supported by the European Union's Horizon 2020 research and innovation programme under grant agreement No 870649, the project Future Migration Scenarios for Europe (FUME; https://futuremigration.eu)

## Population Data Preparation
A file with the population dataset along with key field corresponding to the administrative unit should be included in a *Statistics* folder in csv format. A file with the administrative borders should be included in a *Shapefile* folder in shp format.

## Ancillary Data Preparation
A folder with tha ancillary data should be stored in the parent folder. The GHS layer is essential to initiate the process. Topographic layers may include information about land uses, building features, etc.

## Workflow
A *main* file controls the executed processes for each case study. In the *main* file, major variables should be defined based on the input population data and the desired process needs to be selected. The required variables are the following: the city name, nthe ame of the GHS file, the common key between the population data and the administrative unit, the explored demographic groups. Firstly, at least one of the simple heuristic estimates are to be calculated by either performing the pycnophilactic interpolation or the dasymentric mapping. The execution of that produces the desired input for training the regression model. 
Additional parameters need to be defined for training the regression model, such as the type of the model, the desired training dataset, the input to be used, the number of iterations. 
For each of the above outputs, it is suggested to verify the mass preservation, while if the ground truth data at the target resolution is available the direct evaluation of the results may be lastly executed. It is recommended to execute each step seprately.   

#### 1. CALCULATE SIMPLE HEURISTIC ESTIMATES WITH PYCHNOPHYLACTIC OR DASYMETRIC MAPPING
```
run_Pycno = "yes"
# OR
run_Dasy = "yes"
```
#### 2. TRAIN REGRESSION MODEL ()
```
run_Disaggregation = "yes"
```
##### FURTHER CHOICES NEED TO BE DEFINED, SUCH AS:
##### 2.1. SELECT METHOD/MODEL      
##### 2.2. SELECT DISAGGREGATED METHOD TO BE USED AS INPUT   
##### 2.3. SELECT ANCILLARY DATASET

#### 3. VERIFY MASS PRESERVATION
```
verMassPreserv = "yes"
```
#### 4. EVALUATE RESULTS
```
run_EvaluationGC_ams = "yes"
```
