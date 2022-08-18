import os
import json
import time
import logging

from typing import List
from dataclasses import asdict

import hydra
import openai

from omegaconf import OmegaConf
from openai.error import RateLimitError, InvalidRequestError

from src.config import (
    MetricConfig,
    OutputConfig,
    DatasetConfig,
    GenerationConfig,
    GenerationAPIConfig,
    GenerationFuncConfig,
)

logger = logging.getLogger("apiLogger")

def generate(api_cfg: GenerationAPIConfig, prompts: List[str]):
    API_KEY = os.getenv("OPENAI_API_KEY")
    logger.info(f'using apikey: {API_KEY}')
    openai.api_key = API_KEY

    api_options = asdict(api_cfg)
    try:
        response = openai.Completion.create(
            prompt=prompts,
            **api_options
        )
    except RateLimitError as e:
        logger.debug(e)
        if str(e) == 'You exceeded your current quota, please check your plan and billing details.':
            logger.error('We ran out of tokens :(')
            return
        logger.error('We have hit our rate limit. Sleeping for a minute...')
        time.sleep(60)
        response = generate(api_cfg, prompts)

    return response

def batch_generate(api_cfg: GenerationAPIConfig, prompts: List[str]) -> List[str]:
    predictions = []
    try:
        for i in range(0, len(prompts), 20):
            logger.info(f'Generating answer for prompts {i} through {i+20}')
            response = generate(api_cfg, prompts)
            predictions += [prediction["text"] for prediction in response["choices"]]

    except InvalidRequestError as e:
        logger.critical('Sent an invalid request')
        logger.critical(e)
    
    return predictions

def entry_point(
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
    predictions = {id: prediction for id, prediction in zip(data['id'], raw_predictions)}

    logger.info('Computing metrics ...')
    metrics = metric(
        predictions=predictions,
        ground_truths=data
    )

    output_func(
        dataset=data[dataset.text_col],
        answers=data[dataset.label_col],
        metrics=metrics,
        predictions=predictions
    )

@hydra.main(config_path="../conf", config_name="main", version_base="1.2")
def main(cfg: GenerationConfig):
    cfg = OmegaConf.to_object(cfg)
    logger.info(OmegaConf.to_yaml(cfg))
    hydra.utils.instantiate(cfg)

if __name__ == '__main__':
    main()
