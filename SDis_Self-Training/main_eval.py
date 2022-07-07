from process_eval import process_eval
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description='Parameters for multi-output disaggregation with RF and GB')
    parser.add_argument('--attr_value', nargs='+', required=True, help='Input population groups')
    parser.add_argument('--city', type=str, default="ams", help='City, case study area')
    parser.add_argument('--key', type=str, help='Common key between shp and csv')    
    parser.add_argument('--methodopts', nargs='+', help='Select method of disaggregation (aplm (linear model), aprf (random forest), apcatbr (Catboost Regressor))')
    parser.add_argument('--verMassPreserv', type=str, default='no', help='Verify mass preservation')
    parser.add_argument('--run_Evaluation', type=str, default='no', help='Evaluation of results')
    parser.add_argument('--calc_Metrics', type=str, default='no', help='Evaluation of results')
    parser.add_argument('--calc_Corr', type=str, default='no', help='Evaluation of results')
    parser.add_argument('--plot_Scatter', type=str, default='no', help='Evaluation of results')
    parser.add_argument('--plot_evalMaps', type=str, default='no', help='Evaluation of results')
    parser.add_argument('--calc_Metrics_knn', type=str, default='no', help='Evaluation of results')
    parser.add_argument('--plot_evalMaps_knn', type=str, default='no', help='Evaluation of results')
    parser.add_argument('--plot_Matrices', type=str, default='no', help='Evaluation of results')

    args, unknown = parser.parse_known_args()
    #args = parser.parse_args()
    print(args)
    print('----- Arguments successfully passed -----')

    process_eval(attr_value = args.attr_value, city=args.city, key=args.key, methodopts=args.methodopts, 
                 verMassPreserv=args.verMassPreserv, run_Evaluation= args.run_Evaluation, 
                 calc_Metrics=args.calc_Metrics, calc_Corr=args.calc_Corr, plot_Scatter = args.plot_Scatter, plot_evalMaps=args.plot_evalMaps, 
                 calc_Metrics_knn= args.calc_Metrics_knn, plot_evalMaps_knn=args.plot_evalMaps_knn, 
                 plot_Matrices= args.plot_Matrices)

if __name__ == '__main__':
    main()
    
