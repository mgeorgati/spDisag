from utils import npu

def rundasymmapping(idsdataset, polygonvaluesdataset, ancdataset, rastergeo, tempfileid=None):
    print('| DASYMETRIC MAPPING')
    
    idpolvalues = npu.polygonValuesByID(polygonvaluesdataset, idsdataset)
    ancdatasetvalues = npu.statsByID(ancdataset, idsdataset, 'sum')

    # Divide the true polygon values by the estimated polygon values (= ratio)
    polygonratios = {k: idpolvalues[k] / ancdatasetvalues[k] for k in ancdatasetvalues.keys() & idpolvalues}
    
    # Multiply ratio by the different cells within each polygon
    for polid in polygonratios:
        ancdataset[idsdataset == polid] = (ancdataset[idsdataset == polid] * polygonratios[polid])
    
    return ancdataset, rastergeo
