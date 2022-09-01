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

        if self.shuffle:
            self.dataset = self.dataset.shuffle(seed=self.random_seed)

        if isinstance(self.remove_columns, bool):
            self.remove_columns = self.dataset.column_names if self.remove_columns else []
            if self.label_col in self.remove_columns:
                self.remove_columns.remove(self.label_col)

        self.few_shot_examples = ""
        if self.n_few_shot > 0:
            few_shot_dataset = self.dataset.select(range(self.select_n_samples, self.select_n_samples + self.n_few_shot))
            few_shot_dataset = few_shot_dataset.map(
                lambda x: self._preprocess(x, is_few_shot=True),
                batched=self.batched,
                batch_size=self.batch_size,
                remove_columns=self.remove_columns,
                load_from_cache_file=False,
            )
            self.few_shot_examples = self.few_shot_delim.join(few_shot_dataset[self.text_col])
            self.few_shot_examples += self.few_shot_delim

        self.dataset = self.dataset.select(list(range(self.select_n_samples)))

    def __call__(self) -> datasets.dataset_dict:
        return self.dataset.map(
            self._preprocess,
            batched=self.batched,
            batch_size=self.batch_size,
            remove_columns=self.remove_columns,
            load_from_cache_file=self.load_from_cache,
        )

    def _preprocess(self, examples: BatchType, **kw) -> BatchType:
        examples[self.label_col] = examples[self.label_col]
        return examples

@dataclass
class HotPotQA(BaseDataset, HotPotQAConfig):
    def _preprocess(self, examples: BatchType, is_few_shot: bool=False) -> BatchType:
        sentences_grouped = [
            [''.join(sent) for sent in sent_list['sentences']]
            for sent_list in examples['context']
        ]
        sentences_arr = [
            self.sentence_delim.join(sent) for sent in sentences_grouped
        ]
        header = "" if is_few_shot else self.header
        examples[self.text_col] = [
            header \
            + self.few_shot_examples \
            + self.prefix \
            + sent \
            + self.suffix \
            + examples['question'][i]
            for i, sent in enumerate(sentences_arr)
        ]
        examples = super()._preprocess(examples)
        return examples
