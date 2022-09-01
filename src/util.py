import json
import logging

from typing import Any, Dict, List

from src.config import (
    MetricConfig,
    OutputConfig,
    DatasetConfig,
    GenerationConfig,
    GenerationAPIConfig,
    GenerationFuncConfig,
)

logger = logging.getLogger("apiLogger")

def write_output(
    indent: int,
    input_output_df_fname: str,
    metric_results_out_fname: str,
    predictions_output_fname: str,
    input_dataset_output_fname: str,
    per_question_metrics_out_fname: str,
    dataset: List[str],
    answers: List[str],
    metrics: Dict[str, Any],
    per_question_metrics: Dict[str, Dict[str, float]],
    predictions: Dict[str, str],
):
    def write_json(fname: str, obj: Any, msg: str=""):
        if msg: logger.info(msg)
        with open(fname, 'w') as f:
            json.dump(obj, f, indent=indent)

    if metric_results_out_fname:
        write_json(metric_results_out_fname, metrics, msg='Writing metrics to disk')

    if input_dataset_output_fname:
        write_json(input_dataset_output_fname, dataset, msg="Writing input data to disk")

    if predictions_output_fname:
        write_json(predictions_output_fname, predictions, msg='Writing predictions to disk')
    
    if per_question_metrics_out_fname:
        write_json(per_question_metrics_out_fname, per_question_metrics, msg="Writing per question metrics to disk")

    if input_output_df_fname:
        data = {
            key: {
                "input": input_example,
                "output": output_example,
                "answer": answer_example,
            }
            for input_example, (key, output_example), answer_example 
            in zip(dataset, predictions.items(), answers)
        }
        write_json(input_output_df_fname, data, msg="Writing input output pairs to disk")


def dry_run(api_cfg: GenerationAPIConfig, prompts: List[str]):
    logger.info('Dry run generation. Not calling OpenAI API.')
    return prompts

def unit_test(
    api: GenerationAPIConfig,
    dataset: DatasetConfig,
    metric: MetricConfig,
    output_func: OutputConfig,
    generate_func: GenerationFuncConfig,
):
    # Call the dataset preprocess method
    logger.info(f'I am the dataset features BEFORE preprocessing: {dataset.dataset.features}')
    data = dataset()
    logger.info(f'I am the dataset features AFTER preprocessing: {data.features}')

    logger.info('Running predictions ...')
    raw_predictions = generate_func(api_cfg=api, prompts=data[dataset.text_col])
    predictions = {ex['id']: ex[dataset.label_col] for ex in data}

    logger.info('Computing metrics ...')
    metrics, per_question_metrics = metric(
        predictions=predictions,
        ground_truths=data
    )
    
    output_func(
        dataset=data[dataset.text_col],
        answers=data[dataset.label_col],
        metrics=metrics,
        per_question_metrics=per_question_metrics,
        predictions=predictions
    )
