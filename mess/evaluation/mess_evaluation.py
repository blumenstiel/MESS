
# This script combines the results of from the single datasets for the MESS results
# run script from the root of the project with
# python mess/evaluation/mess_evaluation.py --model_outputs /path/to/model_outputs
# e.g. python mess/evaluation/mess_evaluation.py --model_outputs output/SAN_base output/SAN_large

import argparse
import torch
import pandas as pd
from pathlib import Path

dataset_names = {
    'bdd100k_sem_seg_val': 'BDD100K',
    'dark_zurich_sem_seg_val': 'Dark Zurich',
    'mhp_v1_sem_seg_test': 'MHP v1',
    'foodseg103_sem_seg_test': 'FoodSeg103',
    'atlantis_sem_seg_test': 'ATLANTIS',
    'dram_sem_seg_test': 'DRAM',
    'isaid_sem_seg_val': 'iSAID',
    'isprs_potsdam_sem_seg_test_irrg': 'ISPRS Potsdam',
    'worldfloods_sem_seg_test_irrg': 'WorldFloods',
    'floodnet_sem_seg_test': 'FloodNet',
    'uavid_sem_seg_val': 'UAVid',
    'kvasir_instrument_sem_seg_test': 'Kvasir-Instrument',
    'chase_db1_sem_seg_test': 'CHASE DB1',
    'cryonuseg_sem_seg_test': 'CryoNuSeg',
    'paxray_sem_seg_test_lungs': 'PAXRay-Lungs',
    'paxray_sem_seg_test_bones': 'PAXRay-Bones',
    'paxray_sem_seg_test_mediastinum': 'PAXRay-Mediastinum',
    'paxray_sem_seg_test_diaphragm': 'PAXRay-Diaphragm',
    'paxray_combined': 'PAXRay-4',
    'corrosion_cs_sem_seg_test': 'Corrosion CS',
    'deepcrack_sem_seg_test': 'DeepCrack',
    'pst900_sem_seg_test': 'PST900',
    'zerowaste_sem_seg_test': 'ZeroWaste-f',
    'suim_sem_seg_test': 'SUIM',
    'cub_200_sem_seg_test': 'CUB-200',
    'cwfid_sem_seg_test': 'CWFID',
}

dataset_domains = {
    'mhp_v1_sem_seg_test': 'General',
    'foodseg103_sem_seg_test': 'General',
    'bdd100k_sem_seg_val': 'General',
    'dark_zurich_sem_seg_val': 'General',
    'atlantis_sem_seg_test': 'General',
    'dram_sem_seg_test': 'General',
    'isaid_sem_seg_val': 'Earth Monitoring',
    'isprs_potsdam_sem_seg_test_irrg': 'Earth Monitoring',
    'worldfloods_sem_seg_test_irrg': 'Earth Monitoring',
    'floodnet_sem_seg_test': 'Earth Monitoring',
    'uavid_sem_seg_val': 'Earth Monitoring',
    'kvasir_instrument_sem_seg_test': 'Medical Sciences',
    'chase_db1_sem_seg_test': 'Medical Sciences',
    'cryonuseg_sem_seg_test': 'Medical Sciences',
    'paxray_sem_seg_test_lungs': 'Medical Sciences',
    'paxray_sem_seg_test_bones': 'Medical Sciences',
    'paxray_sem_seg_test_mediastinum': 'Medical Sciences',
    'paxray_sem_seg_test_diaphragm': 'Medical Sciences',
    'paxray_combined': 'Medical Sciences',
    'corrosion_cs_sem_seg_test': 'Engineering',
    'deepcrack_sem_seg_test': 'Engineering',
    'pst900_sem_seg_test': 'Engineering',
    'zerowaste_sem_seg_test': 'Engineering',
    'suim_sem_seg_test': 'Agriculture and Biology',
    'cub_200_sem_seg_test': 'Agriculture and Biology',
    'cwfid_sem_seg_test': 'Agriculture and Biology',
}


def combine_paxray(df, model):
    """
    PAXRay is a multi-class dataset. Each class is predicted separately.
    This function combines the separate results in df for the model into one result PAXRay-4.
    """
    paxray_results = df[(df['Model'] == model) & ((df['Dataset'] == 'PAXRay-Lungs') |
                                                (df['Dataset'] == 'PAXRay-Bones') |
                                                (df['Dataset'] == 'PAXRay-Mediastinum') |
                                                (df['Dataset'] == 'PAXRay-Diaphragm'))]
    if len(paxray_results) != 4:
        print('PAXRay results not found')
        return df

    # Combine results
    dataset_results = pd.DataFrame({
        'Model': [output_dir.name],
        'Dataset': ['PAXRay-4'],
        'Domain': ['Medical Sciences'],
    })
    for metric in args.metrics:
        dataset_results[metric] = [paxray_results[metric].mean()]
    df = pd.concat([df, dataset_results], ignore_index=True)

    # Remove single results
    df.drop(df[(df['Dataset'] == 'PAXRay-Lungs') | (df['Dataset'] == 'PAXRay-Bones') |
                 (df['Dataset'] == 'PAXRay-Mediastinum') | (df['Dataset'] == 'PAXRay-Diaphragm')].index, inplace=True)

    return df


if __name__ == '__main__':
    # Parser
    parser = argparse.ArgumentParser(description='Prepare datasets')
    parser.add_argument('--model_outputs', nargs='+', type=str, help='Directory of to model outputs.'
                                                           'Mulitple models can be specified by separating their '
                                                           'output dirs with a space')
    parser.add_argument('--metrics', nargs='+', default=['mIoU'],
                        help='A list of the evaluation metrics to be used')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    args = parser.parse_args()

    # Initialize results dataframe
    results = pd.DataFrame(columns=['Model', 'Dataset', 'Domain'] + args.metrics)

    # Loop through model outputs
    for model_output in args.model_outputs:
        output_dir = Path(model_output)
        for dataset_id, dataset_name in dataset_names.items():
            if dataset_id == 'paxray_combined':
                # Combine single paxray results
                results = combine_paxray(results, model=output_dir.name)
                continue

            # Initialize dataset results
            dataset_results = pd.DataFrame({
                'Model': [output_dir.name],
                'Dataset': [dataset_name],
                'Domain': [dataset_domains[dataset_id]],
            })

            # Get the metrics file path
            metrics_file = list(output_dir.glob(f'{dataset_id}/**/sem_seg_evaluation.pth'))
            if len(metrics_file) == 0:
                print(f'No evaluation file found for {dataset_id} in {model_output}')
                results = pd.concat([results, dataset_results], ignore_index=True)
                continue

            assert len(metrics_file) == 1, \
                f'Found {len(metrics_file)} evaluation files for {dataset_name} in {model_output}'

            # Get metrics from the file
            metrics = torch.load(metrics_file[0])
            for metric in args.metrics:
                if metric in metrics:
                    dataset_results[metric] = [metrics[metric]]
            results = pd.concat([results, dataset_results], ignore_index=True)

        # Add mean
        mean_results = pd.DataFrame({
            'Model': [output_dir.name],
            'Dataset': ['Mean'],
            'Domain': ['Mean'],
        })
        for metric in args.metrics:
            mean_results[metric] = [results[results['Model'] == output_dir.name][metric].mean()]
        results = pd.concat([results, mean_results], ignore_index=True)

    # Save order of datasets
    datasets_order = [d for d in dataset_names.values() if d in results['Dataset'].values] + ['Mean']

    # Set index to dataset and model
    results.set_index(['Dataset', 'Model'], inplace=True, drop=True)

    # Save results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results.round(2).fillna('-').to_csv(results_dir / 'results.csv')
    for metric in args.metrics:
        metric_results = results.fillna('-').unstack(1)[metric].T
        metric_results = metric_results[datasets_order]
        metric_results.round(2).to_csv(results_dir / f'results_{metric}.csv')

    # Group results by domain
    results_by_domain = results.groupby(['Model', 'Domain'], sort=False).mean().round(2)
    results_by_domain.to_csv(results_dir / 'domain_results.csv')
    for metric in args.metrics:
        domain_metric_results = results_by_domain.fillna('-').unstack(0)[metric].T
        print(f'\n{metric} results for domains')
        print(domain_metric_results)
        domain_metric_results.to_csv(results_dir / f'domain_results_{metric}.csv')

    print(f'\nResults saved to {results_dir}')
