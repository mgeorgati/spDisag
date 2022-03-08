# Spatial Disaggregation
This repository contains the code for the spatial disaggregation of population data from various administrative levels 100m. grid cells.
The followed methods are : single- and multi- output Random Forests and Gradient Boosting with Catboost.

You need to have the following:
1. An environment with the packages included in env.yml.
2. An AncillaryData folder with the desired ancillary data for each case study. GHS, CORINE LAND COVER, ESM etc are examples of ancillary data. 
You need to define them in runDisaggregation.py (raster format).
3. A SDis_Self-Training/Shapefiles/ with the vector layer of the administrative units (shp).
4. A SDis_Self-Training/Statistics/ folder with the aggregated counts respectively (csv).

The main architecture was designed by João Monteiro.
Further information can be found in 'Spatial Disaggregation of Population Subgroups Leveraging Self-Trained Multi-Output Gradient Boosted Regression Trees'

Acknowledgements
This work has been supported by the European Union’s Horizon 2020 research and innovation programme under grant agreement No 870649, the project Future Migration Scenarios for Europe (FUME; https://futuremigration.eu)
