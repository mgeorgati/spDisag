import numpy as np, os
import osgeoutils as osgu
import json
def selectAncDt(city, year, inputDataset, ancillary_path):
    if city == 'cph':
        ancdataset1, rastergeo = osgu.readRaster(os.path.join(ancillary_path, 'GHS/GHS_POP_100_near_cubicsplineWaterIESM_new.tif'))
        ancdataset2 = osgu.readRaster(os.path.join(ancillary_path, 'ESM/{0}_residential.tif'.format(city)))[0]
        ancdataset3 = osgu.readRaster(os.path.join(ancillary_path, 'corine/agric_{0}_CLC_2012_2018.tif'.format(city)))[0]
        
        ancdataset4 = osgu.readRaster(os.path.join(ancillary_path, 'corine/greenSpacesComb_{0}_CLC_2012_2018.tif'.format(city)))[0]
        #ancdataset4A = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{0}_greenSpacesProximity.tif'.format(city)))[0]
        
        ancdataset5 = osgu.readRaster(os.path.join(ancillary_path, 'corine/urbfabr_{0}_CLC_2012_2018.tif'.format(city)))[0]
        ancdataset6 = osgu.readRaster(os.path.join(ancillary_path, 'corine/waterComb_{0}_CLC_2012_2018.tif'.format(city)))[0]
        
        ancdataset7 = osgu.readRaster(os.path.join(ancillary_path, 'corine/indComb_{0}_CLC_2012_2018.tif'.format(city)))[0]
        #ancdataset7A = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{0}_industryProximity.tif'.format(city)))[0]
        
        ancdataset8 = osgu.readRaster(os.path.join(ancillary_path, 'corine/transp_{0}_CLC_2012_2018.tif'.format(city)))[0]
        
        ancdataset9A = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{0}_{1}_busstopscount.tif'.format(year,city)))[0]
        ancdataset10A = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{0}_{1}_stationcount.tif'.format(year,city)))[0]
        ancdataset11A = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{0}_schools.tif'.format(year,city)))[0]
        ancdataset12A = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{0}_culture.tif'.format(year,city)))[0]

        ancdataset13 = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{0}_constryear.tif'.format(year,city)))[0]
        ancdataset14 = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{0}_floors.tif'.format(year,city)))[0]
        ancdataset15 = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/bbr_housing_mean.tif'))[0]
        #'GHS_ESM_corine': '8AIL0', 'GHS_ESM_corine_transp':12AIL1, 'GHS_ESM_corine_transpA': 12AIL2
        if inputDataset == 'AIL0':
            ancdatasets = np.dstack((ancdataset1, ancdataset2, ancdataset3, ancdataset4, ancdataset5, ancdataset6, ancdataset7, ancdataset8))
        elif inputDataset == 'AIL1':
            ancdatasets = np.dstack((ancdataset2, ancdataset3, ancdataset4, ancdataset5, ancdataset6, ancdataset7, ancdataset8))
        elif inputDataset == 'AIL2':
            ancdatasets = np.dstack((ancdataset2, ancdataset3, ancdataset4, ancdataset5,ancdataset6, ancdataset7, ancdataset8, ancdataset9A, ancdataset10A, ancdataset11A, ancdataset12A))
        elif inputDataset == 'AIL3':
            ancdatasets = np.dstack((ancdataset9A, ancdataset10A, ancdataset11A, ancdataset12A))
        elif inputDataset == 'AIL4':
            ancdatasets = np.dstack((ancdataset2, ancdataset4, ancdataset6, ancdataset7, ancdataset8, ancdataset9A, ancdataset10A, ancdataset11A, ancdataset12A, ancdataset13,  ancdataset14,  ancdataset15 ))
        elif inputDataset == 'AIL5':
            ancdatasets = np.dstack((ancdataset9A, ancdataset10A, ancdataset11A, ancdataset12A, ancdataset13,  ancdataset14,  ancdataset15))
        elif inputDataset == 'AIL6':
            ancdatasets = np.dstack((ancdataset2, ancdataset4, ancdataset5,ancdataset6, ancdataset8,ancdataset9A,ancdataset10A,ancdataset11A,ancdataset12A, ancdataset13))
        elif inputDataset == 'AIL7':
            ancdatasets = np.dstack((ancdataset2, ancdataset3, ancdataset4, ancdataset5, ancdataset6, ancdataset7, ancdataset8, ancdataset9A, ancdataset10A, ancdataset11A, ancdataset12A, ancdataset13,  ancdataset14,  ancdataset15 ))
        elif inputDataset == 'AIL8':
            ancdatasets = np.dstack((ancdataset13, ancdataset14, ancdataset15 ))
        elif inputDataset == 'AIL9':
            ancdatasets = np.dstack((ancdataset2, ancdataset13, ancdataset14, ancdataset15 ))
        elif inputDataset == 'AIL10':
            ancdatasets = np.dstack((ancdataset12A, ancdataset13, ancdataset14, ancdataset15 ))
        elif inputDataset == 'AIL11':
            ancdatasets = np.dstack((ancdataset2, ancdataset12A, ancdataset13, ancdataset14, ancdataset15 ))
        elif inputDataset == 'AIL12':
            ancdatasets = np.dstack((ancdataset2, ancdataset6, ancdataset13, ancdataset14, ancdataset15 ))
        
    if city == 'ams':
        pathGHS = os.path.join(ancillary_path, 'GHS', 'GHS_POP_100_near_cubicspline.tif')
        ancdatasetGHS, rastergeo = osgu.readRaster(pathGHS)
        
        pathESM = os.path.join(ancillary_path, 'ESM/{}_residential.tif'.format(city))
        ancdatasetESM = osgu.readRaster(pathESM)[0]
        
        pathC_agr = os.path.join(ancillary_path, 'corine/agric_{}_CLC_2012_2018.tif'.format(city))
        ancdatasetAGR = osgu.readRaster(pathC_agr)[0]
        
        pathC_gs = os.path.join(ancillary_path, 'corine/greenSpaces_{}_CLC_2012_2018.tif'.format(city))
        ancdatasetGS= osgu.readRaster(pathC_gs)[0] 
        pathC_gsp = os.path.join(ancillary_path, 'temp_tif/{}_greenSpacesProximity.tif'.format(city))
        ancdataset4A = osgu.readRaster(pathC_gsp )[0]
        
        pathC_uf = os.path.join(ancillary_path, 'corine/urbfabr_{}_CLC_2012_2018.tif'.format(city))
        ancdatasetUF = osgu.readRaster(pathC_uf)[0]
        
        pathC_w = os.path.join(ancillary_path, 'corine/waterComb_{}_CLC_2012_2018.tif'.format(city))
        ancdatasetW = osgu.readRaster(pathC_w)[0]
        
        pathC_i = os.path.join(ancillary_path, 'corine/industry_{}_CLC_2012_2018.tif'.format(city))
        ancdatasetIN = osgu.readRaster(pathC_i)[0]
        pathC_ip = os.path.join(ancillary_path, 'temp_tif/{}_industryProximity.tif'.format(city))
        ancdatasetINP = osgu.readRaster(pathC_ip)[0]
        
        pathC_tr = os.path.join(ancillary_path, 'corine/transp_{}_CLC_2012_2018.tif'.format(city))
        ancdatasetTR = osgu.readRaster(pathC_tr)[0]
        
        ancdataset9 = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{}_busstopsProximity.tif'.format(city)))[0]
        ancdataset10 = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{}_trainstationsProximity.tif'.format(city)))[0]
        ancdataset11 = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{}_schoolProximity.tif'.format(city)))[0]
        ancdataset12 = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{}_univProximity.tif'.format(city)))[0]
        
        path_bsc = os.path.join(ancillary_path, 'temp_tif/{}_busstopsCount.tif'.format(city))
        ancdatasetBSC = osgu.readRaster(path_bsc)[0]
        path_tsc = os.path.join(ancillary_path, 'temp_tif/{}_trainstationsCount.tif'.format(city))
        ancdatasetTSC = osgu.readRaster(path_tsc)[0]
        path_sc = os.path.join(ancillary_path, 'temp_tif/{}_schoolCount.tif'.format(city))
        ancdatasetSC = osgu.readRaster(path_sc)[0]
        path_uc = os.path.join(ancillary_path, 'temp_tif/{}_univCount.tif'.format(city))
        ancdatasetUC = osgu.readRaster(path_uc)[0]
        
        ancdataset13 = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/apartments_sdate.tif'))[0]
        #ancdataset13A = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/apartments_sdate_fillnodata100.tif'))[0] #THIS IS OTHER SIZE
        path_cy = os.path.join(ancillary_path, 'temp_tif/{}_buildingYear.tif'.format(city))
        ancdatasetCY = osgu.readRaster(path_cy)[0]
        
        path_bv = os.path.join(ancillary_path, 'temp_tif/{}_buildingVolume.tif'.format(city))
        ancdatasetBV = osgu.readRaster(path_bv)[0]
        
        path_bh = os.path.join(ancillary_path, 'temp_tif/{}_buildingHeight.tif'.format(city))
        ancdatasetBH = osgu.readRaster(path_bh)[0]
        
        #'GHS_ESM_corine': '8AIL0', 'GHS_ESM_corine_transp':12AIL1, 'GHS_ESM_corine_transpA': 12AIL2
        if inputDataset == 'AIL0':
            dataList = [ancdatasetGHS, ancdatasetESM, ancdatasetAGR, ancdatasetGS, ancdatasetUF, ancdatasetW, ancdatasetIN, ancdatasetTR]
            ancdatasets = np.dstack((ancdatasetGHS, ancdatasetESM, ancdatasetAGR, ancdatasetGS, ancdatasetUF, ancdatasetW, ancdatasetIN,ancdatasetTR))
        
        elif inputDataset == 'AIL12':
            ancdatasets = np.dstack((ancdatasetESM, ancdataset4A, ancdatasetW, ancdatasetINP, ancdatasetTR, ancdatasetBSC, ancdatasetTSC, ancdatasetSC, ancdatasetUC, ancdatasetCY,  ancdatasetBV,  ancdatasetBH ))
        
    np.array ={"name": inputDataset, "files":dataList}
    print('----- Ancillary Data successfully defined -----')
    with open('logs/ancData_list.txt', 'w') as f:
        json.dump(dataList, f, indent=2)

    with open('logs/ancData_list.txt', 'r') as f:
        dataList = json.load(f)
    
    return ancdatasets, rastergeo
    """ancdataset1, rastergeo = osgu.readRaster(os.path.join(ancillary_path, 'GHS', 'GHS_POP_100_near_cubicspline.tif'))
        ancdataset2 = osgu.readRaster(os.path.join(ancillary_path, 'ESM/{}_residential.tif'.format(city)))[0]
        ancdataset3 = osgu.readRaster(os.path.join(ancillary_path, 'corine/agric_{}_CLC_2012_2018.tif'.format(city)))[0]
        
        ancdataset4 = osgu.readRaster(os.path.join(ancillary_path, 'corine/greenSpaces_{}_CLC_2012_2018.tif'.format(city)))[0] 
        ancdataset4A = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{}_greenSpacesProximity.tif'.format(city)))[0]
        
        ancdataset5 = osgu.readRaster(os.path.join(ancillary_path, 'corine/urbfabr_{}_CLC_2012_2018.tif'.format(city)))[0]
        ancdataset6 = osgu.readRaster(os.path.join(ancillary_path, 'corine/waterComb_{}_CLC_2012_2018.tif'.format(city)))[0]
        
        ancdataset7 = osgu.readRaster(os.path.join(ancillary_path, 'corine/industry_{}_CLC_2012_2018.tif'.format(city)))[0]
        ancdataset7A = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{}_industryProximity.tif'.format(city)))[0]
        
        ancdataset8 = osgu.readRaster(os.path.join(ancillary_path, 'corine/transp_{}_CLC_2012_2018.tif'.format(city)))[0]
        
        ancdataset9 = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{}_busstopsProximity.tif'.format(city)))[0]
        ancdataset10 = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{}_trainstationsProximity.tif'.format(city)))[0]
        ancdataset11 = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{}_schoolProximity.tif'.format(city)))[0]
        ancdataset12 = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{}_univProximity.tif'.format(city)))[0]
        
        ancdataset9A = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{}_busstopsCount.tif'.format(city)))[0]
        ancdataset10A = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{}_trainstationsCount.tif'.format(city)))[0]
        ancdataset11A = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{}_schoolCount.tif'.format(city)))[0]
        ancdataset12A = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{}_univCount.tif'.format(city)))[0]
        
        ancdataset13 = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/apartments_sdate.tif'))[0]
        ancdataset13A = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/apartments_sdate_fillnodata100.tif'))[0] #THIS IS OTHER SIZE
        ancdataset13B = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{}_buildingYear.tif'.format(city)))[0]
        
        ancdataset14 = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{}_buildingVolume.tif'.format(city)))[0]
        ancdataset15 = osgu.readRaster(os.path.join(ancillary_path, 'temp_tif/{}_buildingHeight.tif'.format(city)))[0]
        
        #'GHS_ESM_corine': '8AIL0', 'GHS_ESM_corine_transp':12AIL1, 'GHS_ESM_corine_transpA': 12AIL2
        if inputDataset == 'AIL0':
            ancdatasets = np.dstack((ancdataset1, ancdataset2, ancdataset3, ancdataset4, ancdataset5,ancdataset6, ancdataset7,ancdataset8))
        elif inputDataset == 'AIL1':
            ancdatasets = np.dstack((ancdataset1, ancdataset2, ancdataset3, ancdataset4, ancdataset5,ancdataset6, ancdataset7, ancdataset8,ancdataset9,ancdataset10,ancdataset11,ancdataset12))
        elif inputDataset == 'AIL2':
            ancdatasets = np.dstack((ancdataset1, ancdataset2, ancdataset3, ancdataset4, ancdataset5,ancdataset6, ancdataset7, ancdataset8,ancdataset9A,ancdataset10A,ancdataset11A,ancdataset12A))
        elif inputDataset == 'AIL3':
            ancdatasets = np.dstack((ancdataset2, ancdataset3, ancdataset4, ancdataset5,ancdataset6, ancdataset7, ancdataset8))
        elif inputDataset == 'AIL4':
            ancdatasets = np.dstack((ancdataset2, ancdataset3, ancdataset4, ancdataset5,ancdataset6, ancdataset7, ancdataset8, ancdataset9A,ancdataset10A,ancdataset11A,ancdataset12A, ancdataset13))
        elif inputDataset == 'AIL5':
            ancdatasets = np.dstack((ancdataset13B, ancdataset14, ancdataset15))
        elif inputDataset == 'AIL6':
            ancdatasets = np.dstack((ancdataset2, ancdataset9A,ancdataset10A,ancdataset11A,ancdataset12A, ancdataset13A))
        elif inputDataset == 'AIL7':
            ancdatasets = np.dstack((ancdataset2, ancdataset4, ancdataset5,ancdataset6, ancdataset8,ancdataset9A,ancdataset10A,ancdataset11A,ancdataset12A, ancdataset13))
        elif inputDataset == 'AIL8':
            ancdatasets = np.dstack((ancdataset1,ancdataset2, ancdataset3, ancdataset4, ancdataset5,ancdataset6, ancdataset7, ancdataset8,ancdataset9A,ancdataset10A,ancdataset11A,ancdataset12A, ancdataset13))
        elif inputDataset == 'AIL9':
            ancdatasets = np.dstack((ancdataset1,ancdataset2, ancdataset3, ancdataset4, ancdataset5,ancdataset6, ancdataset7, ancdataset8, ancdataset9A,ancdataset10A,ancdataset11A,ancdataset12A, ancdataset13B))
        elif inputDataset == 'AIL10':
            ancdatasets = np.dstack((ancdataset2, ancdataset3, ancdataset4A, ancdataset5, ancdataset6, ancdataset7A, ancdataset8, ancdataset9A, ancdataset10A,ancdataset11A,ancdataset12A, ancdataset13B))
        elif inputDataset == 'AIL11':
            ancdatasets = np.dstack((ancdataset2, ancdataset3, ancdataset4A, ancdataset5,ancdataset6, ancdataset7A, ancdataset8,ancdataset9A,ancdataset10A,ancdataset11A,ancdataset12A, ancdataset13B,  ancdataset14,  ancdataset15 ))
        elif inputDataset == 'AIL12':
            ancdatasets = np.dstack((ancdataset2, ancdataset4A, ancdataset6, ancdataset7A, ancdataset8,ancdataset9A,ancdataset10A,ancdataset11A,ancdataset12A, ancdataset13B,  ancdataset14,  ancdataset15 ))
        elif inputDataset == 'AIL13':
            ancdatasets = np.dstack((ancdataset2, ancdataset4A, ancdataset5, ancdataset6, ancdataset7A, ancdataset8,ancdataset9A,ancdataset10A,ancdataset11A,ancdataset12A, ancdataset13B,  ancdataset14,  ancdataset15 ))
        
        ######################################
        elif inputDataset == 'AIL1':
            ancdatasets = np.dstack((ancdatasetGHS, ancdatasetESM, ancdatasetAGR, ancdatasetGS, ancdatasetUF,ancdatasetW, ancdatasetIN, ancdataset8,ancdataset9,ancdataset10,ancdataset11,ancdataset12))
        elif inputDataset == 'AIL2':
            ancdatasets = np.dstack((ancdatasetGHS, ancdatasetESM, ancdatasetAGR, ancdatasetGS, ancdatasetUF,ancdatasetW, ancdatasetIN, ancdataset8,ancdataset9A,ancdataset10A,ancdataset11A,ancdataset12A))
        elif inputDataset == 'AIL3':
            ancdatasets = np.dstack((ancdatasetESM, ancdatasetAGR, ancdatasetGS, ancdatasetUF,ancdatasetW, ancdatasetIN, ancdataset8))
        elif inputDataset == 'AIL4':
            ancdatasets = np.dstack((ancdatasetESM, ancdatasetAGR, ancdatasetGS, ancdatasetUF,ancdatasetW, ancdatasetIN, ancdataset8, ancdataset9A,ancdataset10A,ancdataset11A,ancdataset12A, ancdataset13))
        elif inputDataset == 'AIL5':
            ancdatasets = np.dstack((ancdataset13B, ancdatasetBV, ancdatasetBH))
        elif inputDataset == 'AIL6':
            ancdatasets = np.dstack((ancdatasetESM, ancdataset9A,ancdataset10A,ancdataset11A,ancdataset12A, ancdataset13A))
        elif inputDataset == 'AIL7':
            ancdatasets = np.dstack((ancdatasetESM, ancdatasetGS, ancdatasetUF,ancdatasetW, ancdataset8,ancdataset9A,ancdataset10A,ancdataset11A,ancdataset12A, ancdataset13))
        elif inputDataset == 'AIL8':
            ancdatasets = np.dstack((ancdatasetGHS,ancdatasetESM, ancdatasetAGR, ancdatasetGS, ancdatasetUF,ancdatasetW, ancdatasetIN, ancdataset8,ancdataset9A,ancdataset10A,ancdataset11A,ancdataset12A, ancdataset13))
        elif inputDataset == 'AIL9':
            ancdatasets = np.dstack((ancdatasetGHS,ancdatasetESM, ancdatasetAGR, ancdatasetGS, ancdatasetUF,ancdatasetW, ancdatasetIN, ancdataset8, ancdataset9A,ancdataset10A,ancdataset11A,ancdataset12A, ancdataset13B))
        elif inputDataset == 'AIL10':
            ancdatasets = np.dstack((ancdatasetESM, ancdatasetAGR, ancdataset4A, ancdatasetUF, ancdatasetW, ancdatasetINP, ancdataset8, ancdataset9A, ancdataset10A,ancdataset11A,ancdataset12A, ancdataset13B))
        elif inputDataset == 'AIL11':
            ancdatasets = np.dstack((ancdatasetESM, ancdatasetAGR, ancdataset4A, ancdatasetUF, ancdatasetW, ancdatasetINP, ancdataset8,ancdataset9A,ancdataset10A,ancdataset11A,ancdataset12A, ancdataset13B,  ancdatasetBV,  ancdatasetBH ))
     """