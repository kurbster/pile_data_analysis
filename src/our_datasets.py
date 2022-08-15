#!/usr/bin/env python3
from typing import Dict, List, Union
from dataclasses import dataclass

import datasets
from datasets import load_dataset

from src.config import DatasetConfig, HotPotQAConfig

BatchType = Dict[str, Union[str, List[str]]]

@dataclass
class BaseDataset(DatasetConfig):
    _target_: str = ""

    def __post_init__(self):
        self.dataset = load_dataset(
            self.name, self.option,
            split=self.split,
            cache_dir=self.cache_dir
        )

        if isinstance(self.remove_columns, bool):
            self.remove_columns = self.dataset.column_names if self.remove_columns else []
            if self.label_col in self.remove_columns:
                self.remove_columns.remove(self.label_col)

    def __call__(self) -> datasets.dataset_dict:
        return self.dataset.map(
            self._preprocess,
            batched=self.batched,
            batch_size=self.batch_size,
            remove_columns=self.remove_columns,
            load_from_cache_file=self.load_from_cache,
        )

    def _preprocess(self, examples: BatchType) -> BatchType:
        examples[self.label_col] = examples[self.label_col]
        return examples

@dataclass
class HotPotQA(BaseDataset, HotPotQAConfig):
    def _preprocess(self, examples: BatchType) -> BatchType:
        sentences_grouped = [
            [''.join(sent) for sent in sent_list['sentences']]
            for sent_list in examples['context']
        ]
        sentences_arr = [
            self.sentence_delim.join(sent) for sent in sentences_grouped
        ]
        examples[self.text_col] = [
            self.prefix \
            + sent \
            + self.suffix \
            + examples['question'][i]
            for i, sent in enumerate(sentences_arr)
        ]
        examples = super()._preprocess(examples)
        return examples
