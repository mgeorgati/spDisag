import numpy as np, os
from utils import osgu
import json

def selectAncDt(city, year, inputDataset, ancillary_path, pop_path):
    print(city, year, inputDataset, ancillary_path)
    
    if city == 'cph':
        pathGHS = os.path.join(ancillary_path, 'GHS', 'GHS_POP_100_near_cubicspline.tif')
        ancdatasetGHS, rastergeo = osgu.readRaster(pathGHS)
        
        pathESM = os.path.join(ancillary_path, 'ESM/{}_residential.tif'.format(city))
        ancdatasetESM = osgu.readRaster(pathESM)[0]
        
        pathC_agr = os.path.join(ancillary_path, 'corine/agric_{}_CLC_2012_2018.tif'.format(city))
        ancdatasetAGR = osgu.readRaster(pathC_agr)[0]
        
        pathC_gs = os.path.join(ancillary_path, 'corine/greenSpaces_{}_CLC_2012_2018.tif'.format(city))
        ancdatasetGS= osgu.readRaster(pathC_gs)[0] 
        pathC_gsp = os.path.join(ancillary_path, 'corine/greenSpacesComb_{}_CLC_2012_2018.tif'.format(city))
        ancdatasetGSP = osgu.readRaster(pathC_gsp )[0]
        
        pathC_uf = os.path.join(ancillary_path, 'corine/urbfabr_{}_CLC_2012_2018.tif'.format(city))
        ancdatasetUF = osgu.readRaster(pathC_uf)[0]
        
        pathC_w = os.path.join(ancillary_path, 'corine/waterComb_{}_CLC_2012_2018.tif'.format(city))
        ancdatasetW = osgu.readRaster(pathC_w)[0]
        
        pathC_i = os.path.join(ancillary_path, 'corine/industry_{}_CLC_2012_2018.tif'.format(city))
        ancdatasetIN = osgu.readRaster(pathC_i)[0]
        pathC_ip = os.path.join(ancillary_path, 'corine/indComb_{}_CLC_2012_2018.tif'.format(city))
        ancdatasetINP = osgu.readRaster(pathC_ip)[0]
        
        pathC_tr = os.path.join(ancillary_path, 'corine/transp_{}_CLC_2012_2018.tif'.format(city))
        ancdatasetTR = osgu.readRaster(pathC_tr)[0]
        
        path_bsc = os.path.join(ancillary_path, 'temp_tif/{0}_{1}_busstopscount.tif'.format(year,city))
        ancdatasetBSC = osgu.readRaster(path_bsc)[0]
        path_tsc = os.path.join(ancillary_path, 'temp_tif/{0}_{1}_stationcount.tif'.format(year,city))
        ancdatasetTSC = osgu.readRaster(path_tsc)[0]
        path_sc = os.path.join(ancillary_path, 'temp_tif/{}_schools.tif'.format(year))
        ancdatasetSC = osgu.readRaster(path_sc)[0]
        path_uc = os.path.join(ancillary_path, 'temp_tif/{}_culture.tif'.format(year))
        ancdatasetUC = osgu.readRaster(path_uc)[0]

        path_cy  = os.path.join(ancillary_path, 'temp_tif/{0}_constryear.tif'.format(year,city))
        ancdatasetCY = osgu.readRaster(path_cy)[0]
        path_bh = os.path.join(ancillary_path, 'temp_tif/{0}_floors.tif'.format(year,city))
        ancdatasetBH = osgu.readRaster(path_bh)[0]
        path_hp = os.path.join(ancillary_path, 'temp_tif/bbr_housing_mean.tif')
        ancdatasetHP = osgu.readRaster(path_hp)[0]
        """
        path_totalpop = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l1_sum_population.tif'.format(city))
        ancdatasetTP = osgu.readRaster(path_totalpop)[0]
        
        path_agpop1 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l2_p00_19.tif'.format(city))
        ancdatasetAG1 = osgu.readRaster(path_agpop1)[0]
        path_agpop2 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l3_p20_29.tif'.format(city))
        ancdatasetAG2 = osgu.readRaster(path_agpop2)[0]
        path_agpop3 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l4_p30_44.tif'.format(city))
        ancdatasetAG3 = osgu.readRaster(path_agpop3)[0]
        path_agpop4 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l5_p45_64.tif'.format(city))
        ancdatasetAG4 = osgu.readRaster(path_agpop4)[0]
        path_agpop5 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l6_p65.tif'.format(city))
        ancdatasetAG5 = osgu.readRaster(path_agpop5)[0]

        path_mbpop0 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l7_p_immig.tif'.format(city))
        ancdatasetMB0 = osgu.readRaster(path_mbpop0)[0]
        path_mbpop1 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l8_p_eumig.tif'.format(city))
        ancdatasetMB1 = osgu.readRaster(path_mbpop1)[0]
        path_mbpop2 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l9_p_noeumig.tif'.format(city))
        ancdatasetMB2 = osgu.readRaster(path_mbpop2)[0]

        path_D1 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l11_sum_births.tif'.format(city))
        ancdatasetD1 = osgu.readRaster(path_D1)[0]
        path_D2 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l12_sum_deaths.tif'.format(city))
        ancdatasetD2 = osgu.readRaster(path_D2)[0]
        path_D3 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l13a_sum_marriages.tif'.format(city))
        ancdatasetD3 = osgu.readRaster(path_D3)[0]
        path_D4 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l14a_p_outmigr.tif'.format(city))
        ancdatasetD4 = osgu.readRaster(path_D4)[0]
        path_D5 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l14b_p_outmigr_nodk.tif'.format(city))
        ancdatasetD5 = osgu.readRaster(path_D5)[0]
        path_D6 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l15a_p_inmigr.tif'.format(city))
        ancdatasetD6 = osgu.readRaster(path_D6)[0]
        path_D7 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l15b_p_inmigr_nodk.tif'.format(city))
        ancdatasetD7 = osgu.readRaster(path_D7)[0]
        path_D8 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l16_p_outmigr_incph.tif'.format(city))
        ancdatasetD8 = osgu.readRaster(path_D8)[0]
        path_D9 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l17_p_inmigr_incph.tif'.format(city))
        ancdatasetD9 = osgu.readRaster(path_D9)[0]
        path_D10 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l18_p_teduc.tif'.format(city))
        ancdatasetD10 = osgu.readRaster(path_D10)[0]
        path_D11 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l19_p_rich_new.tif'.format(city))
        ancdatasetD11 = osgu.readRaster(path_D11)[0]
        path_D12 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l20_p_poor_new.tif'.format(city))
        ancdatasetD12 = osgu.readRaster(path_D12)[0]

        path_build1 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l21_p_notused.tif'.format(city))
        ancdatasetBL1 = osgu.readRaster(path_build1)[0]
        path_build2 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l22_p_rented.tif'.format(city))
        ancdatasetBL2 = osgu.readRaster(path_build2)[0]
        path_build3 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l23_p_private.tif'.format(city))
        ancdatasetBL3 = osgu.readRaster(path_build3)[0]
        path_build4 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l24_p_public.tif'.format(city))
        ancdatasetBL4 = osgu.readRaster(path_build4)[0]
        path_build5 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l25_p_area.tif'.format(city))
        ancdatasetBL5 = osgu.readRaster(path_build5)[0]
        path_build6 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l26_p_home.tif'.format(city))
        ancdatasetBL6 = osgu.readRaster(path_build6)[0]
        path_build7 = os.path.join(pop_path, 'PopData/2008/temp_tif/demo/2008_l27_p_nrooms.tif'.format(city))
        ancdatasetBL7 = osgu.readRaster(path_build7)[0] """
        

        if inputDataset == 'AIL8':
            dataList = [path_cy, path_bh, path_hp]        
            ancdatasets = np.dstack((ancdatasetCY, ancdatasetBH , ancdatasetHP))
        elif inputDataset == 'AIL50':
            dataList = [pathGHS, pathESM, pathC_agr, pathC_gs, pathC_gsp, pathC_uf, pathC_w, pathC_i, pathC_ip, pathC_tr,                       
                        path_bsc, path_tsc, path_sc, path_uc,
                        path_cy, path_hp, path_bh,
                        path_totalpop, path_agpop1, path_agpop2, path_agpop3, path_agpop4, path_agpop5,
                        path_mbpop0, path_mbpop1, path_mbpop2, path_build1]
            
            ancdatasets = np.dstack((ancdatasetGHS, ancdatasetESM, ancdatasetAGR, ancdatasetGS, ancdatasetGSP, ancdatasetUF, ancdatasetW, ancdatasetIN, ancdatasetINP, ancdatasetTR,
                                    ancdatasetBSC, ancdatasetTSC, ancdatasetSC,ancdatasetUC,
                                    ancdatasetCY, ancdatasetHP, ancdatasetBH,
                                    ancdatasetTP, ancdatasetAG1, ancdatasetAG2, ancdatasetAG3,ancdatasetAG4,ancdatasetAG5,
                                    ancdatasetMB0, ancdatasetMB1, ancdatasetMB2, ancdatasetBL1))
        elif inputDataset == 'AIL51':
            dataList = [ pathESM, pathC_agr, pathC_gs, pathC_gsp, pathC_uf, pathC_w, pathC_i, pathC_ip, pathC_tr,
                        path_bsc, path_tsc, path_sc, path_uc,
                        path_cy, path_hp, path_bh,
                        path_totalpop, path_agpop1, path_agpop2, path_agpop3, path_agpop4, path_agpop5,
                        path_mbpop0, path_mbpop1, path_mbpop2, 
                        path_D1,path_D2,path_D3,path_D4,path_D5,path_D6,path_D7,path_D8,path_D9,path_D10,path_D11,path_D12,
                        path_build1, path_build2,path_build3,path_build4,path_build5,path_build6,path_build7]
            
            ancdatasets = np.dstack(( ancdatasetESM, ancdatasetAGR, ancdatasetGS, ancdatasetGSP, ancdatasetUF, ancdatasetW, ancdatasetIN, ancdatasetINP, ancdatasetTR,
                                    ancdatasetBSC, ancdatasetTSC, ancdatasetSC,ancdatasetUC,
                                    ancdatasetCY, ancdatasetHP, ancdatasetBH,
                                    ancdatasetTP, ancdatasetAG1, ancdatasetAG2, ancdatasetAG3,ancdatasetAG4,ancdatasetAG5,
                                    ancdatasetMB0, ancdatasetMB1, ancdatasetMB2, 
                                    ancdatasetD1,ancdatasetD2,ancdatasetD3,ancdatasetD4,ancdatasetD5,ancdatasetD6,ancdatasetD7,ancdatasetD8,ancdatasetD9,ancdatasetD10,ancdatasetD11,ancdatasetD12,
                                    ancdatasetBL1,ancdatasetBL2,ancdatasetBL3,ancdatasetBL4,ancdatasetBL5,ancdatasetBL6,ancdatasetBL7))
        elif inputDataset == 'AIL52':
            dataList = [ pathESM, pathC_gsp, pathC_uf, pathC_w, pathC_i, pathC_ip, 
                        path_bsc, path_tsc, path_sc, path_uc,
                        path_cy, path_hp, path_bh,
                        path_totalpop, #path_agpop1, path_agpop2, path_agpop3, path_agpop4, path_agpop5,
                        path_mbpop0, path_mbpop1, path_mbpop2, 
                        #path_D4,path_D6,path_D8,path_D10,path_D11,path_D12,
                        path_build1, path_build2,path_build3,path_build5,path_build6,path_build7]
            
            ancdatasets = np.dstack(( ancdatasetESM, ancdatasetGSP, ancdatasetUF, ancdatasetW, ancdatasetIN, ancdatasetINP, 
                                    ancdatasetBSC, ancdatasetTSC, ancdatasetSC,ancdatasetUC,
                                    ancdatasetCY, ancdatasetHP, ancdatasetBH,
                                    ancdatasetTP, #ancdatasetAG1, ancdatasetAG2, ancdatasetAG3,ancdatasetAG4,ancdatasetAG5,
                                    ancdatasetMB0, ancdatasetMB1, ancdatasetMB2, 
                                    #ancdatasetD4,ancdatasetD6,ancdatasetD8,ancdatasetD10,ancdatasetD11,ancdatasetD12,
                                    ancdatasetBL1,ancdatasetBL2,ancdatasetBL3,ancdatasetBL5,ancdatasetBL6,ancdatasetBL7))
        
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
        ancdatasetGSP = osgu.readRaster(pathC_gsp )[0]
        
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
        
        path_bsc1 = os.path.join(ancillary_path, 'temp_tif/{}_busstopsProximity.tif'.format(city))
        ancdatasetBSC1 = osgu.readRaster(path_bsc1)[0]
        path_tsc1 = os.path.join(ancillary_path, 'temp_tif/{}_trainstationsProximity.tif'.format(city))
        ancdatasetTSC1 = osgu.readRaster(path_tsc1)[0]
        path_sc1 = os.path.join(ancillary_path, 'temp_tif/{}_schoolProximity.tif'.format(city))
        ancdatasetSC1 = osgu.readRaster(path_sc1)[0]
        path_uc1 = os.path.join(ancillary_path, 'temp_tif/{}_univProximity.tif'.format(city))
        ancdatasetUC1 = osgu.readRaster(path_uc1)[0]
        
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

        path_totalpop = os.path.join(ancillary_path, 'temp_tif/2008_l1_totalpop.tif'.format(city))
        ancdatasetTP = osgu.readRaster(path_totalpop)[0]
        
        path_agpop1 = os.path.join(ancillary_path, 'temp_tif/2008_l2_children.tif'.format(city))
        ancdatasetAG1 = osgu.readRaster(path_agpop1)[0]
        path_agpop2 = os.path.join(ancillary_path, 'temp_tif/2008_l3_students.tif'.format(city))
        ancdatasetAG2 = osgu.readRaster(path_agpop2)[0]
        path_agpop3 = os.path.join(ancillary_path, 'temp_tif/2008_l4_mobile_adults.tif'.format(city))
        ancdatasetAG3 = osgu.readRaster(path_agpop3)[0]
        path_agpop4 = os.path.join(ancillary_path, 'temp_tif/2008_l5_not_mobile_adults.tif'.format(city))
        ancdatasetAG4 = osgu.readRaster(path_agpop4)[0]
        path_agpop5 = os.path.join(ancillary_path, 'temp_tif/2008_l6_elderly.tif'.format(city))
        ancdatasetAG5 = osgu.readRaster(path_agpop5)[0]

        path_mbpop0 = os.path.join(ancillary_path, 'temp_tif/2008_l7_immigrants.tif'.format(city))
        ancdatasetMB0 = osgu.readRaster(path_mbpop0)[0]
        path_mbpop1 = os.path.join(ancillary_path, 'temp_tif/2008_l8_eu_immigrants.tif'.format(city))
        ancdatasetMB1 = osgu.readRaster(path_mbpop1)[0]
        path_mbpop2 = os.path.join(ancillary_path, 'temp_tif/2008_l9_noneu_immigrants.tif'.format(city))
        ancdatasetMB2 = osgu.readRaster(path_mbpop2)[0]

        path_build1 = os.path.join(ancillary_path, 'temp_tif/2008_l25_total_area_of_residence.tif'.format(city))
        ancdatasetBL1 = osgu.readRaster(path_build1)[0]
        
        #'GHS_ESM_corine': '8AIL0', 'GHS_ESM_corine_transp':12AIL1, 'GHS_ESM_corine_transpA': 12AIL2
        if inputDataset == 'AIL0':
            ancdatasets = np.dstack((ancdatasetGHS, ancdatasetESM, ancdatasetAGR, ancdatasetGS, ancdatasetUF, ancdatasetW, ancdatasetIN,ancdatasetTR))
        
        elif inputDataset == 'AIL12':
            dataList = [pathESM, pathC_gsp, pathC_w,  pathC_ip, pathC_tr, path_bsc, path_tsc, path_sc, path_uc, path_cy, path_bv, path_bh]
            ancdatasets = np.dstack((ancdatasetESM, ancdatasetGSP, ancdatasetW, ancdatasetINP, ancdatasetTR, ancdatasetBSC, ancdatasetTSC, ancdatasetSC, ancdatasetUC, ancdatasetCY,  ancdatasetBV,  ancdatasetBH ))
        
        elif inputDataset == 'AIL20':
            dataList = [pathESM, path_build1]
            ancdatasets = np.dstack((ancdatasetESM, ancdatasetBL1))
        elif inputDataset == 'AIL21':
            dataList = [path_totalpop, path_mbpop0]
            ancdatasets = np.dstack((ancdatasetTP, ancdatasetMB0))
        elif inputDataset == 'AIL22':
            dataList = [path_totalpop, path_mbpop0, path_build1]
            ancdatasets = np.dstack((ancdatasetTP, ancdatasetMB0, ancdatasetBL1))
        elif inputDataset == 'AIL23':
            dataList = [path_totalpop, path_agpop1, path_agpop2, path_agpop3, path_agpop4, path_agpop5]
            ancdatasets = np.dstack((ancdatasetTP, ancdatasetAG1, ancdatasetAG2, ancdatasetAG3, ancdatasetAG4, ancdatasetAG5))
        elif inputDataset == 'AIL24':
            dataList = [path_totalpop, path_build1]
            ancdatasets = np.dstack((ancdatasetTP, ancdatasetBL1))
        elif inputDataset == 'AIL50':
            dataList = [pathGHS, pathESM, pathC_agr, pathC_gs, pathC_gsp, pathC_uf, pathC_w, pathC_i, pathC_ip, pathC_tr,
                        path_bsc1, path_tsc1, path_sc1, path_uc1,
                        path_bsc, path_tsc, path_sc, path_uc,
                        path_cy, path_bv, path_bh,
                        path_totalpop, path_agpop1, path_agpop2, path_agpop3, path_agpop4, path_agpop5,
                        path_mbpop0, path_mbpop1, path_mbpop2, path_build1]
            
            ancdatasets = np.dstack((ancdatasetGHS, ancdatasetESM, ancdatasetAGR, ancdatasetGS, ancdatasetGSP, ancdatasetUF, ancdatasetW, ancdatasetIN, ancdatasetINP, ancdatasetTR,
                                    ancdatasetBSC1, ancdatasetTSC1, ancdatasetSC1,ancdatasetUC1,
                                    ancdatasetBSC, ancdatasetTSC, ancdatasetSC,ancdatasetUC,
                                    ancdatasetCY, ancdatasetBV, ancdatasetBH,
                                    ancdatasetTP, ancdatasetAG1, ancdatasetAG2, ancdatasetAG3,ancdatasetAG4,ancdatasetAG5,
                                    ancdatasetMB0, ancdatasetMB1, ancdatasetMB2, ancdatasetBL1))
        
    
    array ={"name": inputDataset, "files": dataList}
    print('----- Ancillary Data successfully defined -----')
    with open('logs/{}_ancData_list.txt'.format(city), 'w') as f:
        json.dump(array, f, indent=2)

    with open('logs/{}_ancData_list.txt'.format(city), 'r') as f:
        array = json.load(f)
    
    return ancdatasets, rastergeo