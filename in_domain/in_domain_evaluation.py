
# This script combines the results of from the single datasets for the MESS results
# run script from the root of the project with
# python in_domain/in_domain_evaluation.py --model_outputs /path/to/model_outputs
# e.g. python in_domain/in_domain_evaluation.py --model_outputs output/SAN_base output/SAN_large

import argparse
import torch
import pandas as pd
from pathlib import Path

dataset_names = {
    'ade20k_sem_seg_val': 'ADE20K-150',
    'ade20k_full_sem_seg_val': 'ADE20K-847',
    'pascal_context_59_sem_seg_val': 'Pascal Context-59',
    'pascal_context_459_sem_seg_val': 'Pascal Context-459',
    'voc_2012_sem_seg_val_bg': 'Pascal VOC',
}


if __name__ == '__main__':
    # Parser
    parser = argparse.ArgumentParser(description='Evaluate in-domain datasets')
    parser.add_argument('--model_outputs', nargs='+', type=str, help='Directory of to model outputs.'
                                                           'Mulitple models can be specified by separating their '
                                                           'output dirs with a space')
    parser.add_argument('--metrics', nargs='+', default=['mIoU', 'CoI-mIoU'],
                        help='A list of the evaluation metrics to be used')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    args = parser.parse_args()

    # Initialize results dataframe
    results = pd.DataFrame(columns=['Model', 'Dataset'] + args.metrics)

    # Loop through model outputs
    for model_output in args.model_outputs:
        output_dir = Path(model_output)
        for dataset_id, dataset_name in dataset_names.items():

            # Initialize dataset results
            dataset_results = pd.DataFrame({
                'Model': [output_dir.name],
                'Dataset': [dataset_name],
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

    results.round(2).fillna('-').to_csv(results_dir / 'in_domain_results.csv')
    for metric in args.metrics:
        metric_results = results.fillna('-').unstack(1)[metric].T
        metric_results = metric_results[datasets_order]
        metric_results.round(2).to_csv(results_dir / f'in_domain_results_{metric}.csv')

    print(f'\nIn-domain results saved to {results_dir}')
