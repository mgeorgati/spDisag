from process import process_data
import numpy as np
import argparse
def main():
    parser = argparse.ArgumentParser(description='Parameters for multi-output disaggregation with RF and GB')
    parser.add_argument('--attr_value', nargs='+', required=True,
                        default=['children', 'students','mobadults', 'nmobadults', 'elderly', 'sur', 'ant', 'mar','tur', 'nonwestern', 'western', 'autoch' ], help='Input population groups')
    parser.add_argument('--city', type=str,
                        default="ams", help='City, case study area')
    parser.add_argument('--popraster', type=str,
                        default='GHS_POP_100_near_cubicspline.tif', help='GHS input layer')
    parser.add_argument('--key', type=str,
                        default='Buurtcode', help='Common key between shp and csv')
    parser.add_argument('--run_Pycno', type=str, default='no', help='Run pycnophylactic interpolation')
    parser.add_argument('--run_Dasy', type=str, default='no', help='Run dasymetric mapping')
    parser.add_argument('--run_Disaggregation', type=str, default='no', help='Run disaggregation')
    parser.add_argument('--maxIters', type=int, default=2, help='Max Iterations')
    parser.add_argument('--methodopts', nargs='+', help='Select method of disaggregation (aplm (linear model), aprf (random forest), apcatbr (Catboost Regressor))')
    #choices = ['aplm', 'aprf', 'apcatbr']
    parser.add_argument('--ymethodopts',  nargs='+', help='Input layers')
    parser.add_argument('--inputDataset', nargs='+', help='Training dataset')
    parser.add_argument('--verMassPreserv', type=str, default='no', help='Verify mass preservation')
    parser.add_argument('--run_Evaluation', type=str, default='no', help='Evaluation of results')

    args, unknown = parser.parse_known_args()
    #args = parser.parse_args()

    process_data(attr_value=args.attr_value, city=args.city, popraster = args.popraster, key=args.key, 
            run_Pycno=args.run_Pycno, run_Dasy=args.run_Dasy, run_Disaggregation = args.run_Disaggregation, maxIters = args.maxIters, methodopts=args.methodopts, ymethodopts=args.ymethodopts, 
            inputDataset=args.inputDataset, verMassPreserv=args.verMassPreserv, run_Evaluation=args.run_Evaluation)

if __name__ == '__main__':
    main()
    
"""

# Save cluster output
print(cluster_assignment)
np.savetxt(args.oname, cluster_assignment, fmt='%d', delimiter=',')

# Save MRF as npy
for key, value in cluster_MRFs.items():
    with open(f'output_folder/MRF_{args.fname.split(".")[0]}_{key}.npy', 'wb') as f:
        np.save(f, np.array(value))
"""