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

        # If remove columns is a bool and True the remove all except label col
        # If False then remove nothing. If not a bool then remove the list
        # of columns provided
        if isinstance(self.remove_columns, bool):
            self.remove_columns = self.dataset.column_names if self.remove_columns else []
            if self.label_col in self.remove_columns:
                self.remove_columns.remove(self.label_col)

        # If no few shot examples are passed then generate them
        if self.n_few_shot > 0 and len(self.few_shot_examples) == 0:
            few_shot_dataset = self.dataset.select(range(self.select_n_samples, self.select_n_samples + self.n_few_shot))
            few_shot_dataset = few_shot_dataset.map(
                self._create_few_shot_examples,
                batched=self.batched,
                batch_size=self.n_few_shot,
                remove_columns=self.remove_columns,
                load_from_cache_file=False,
            )
            self.few_shot_examples = few_shot_dataset[self.text_col]

        self.few_shot_examples = self.section_delim.join(self.few_shot_examples)

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

    def _create_few_shot_examples(self, examples: BatchType) -> BatchType:
        raise NotImplementedError("You must override _create_few_shot_examples for your class!")

@dataclass
class HotPotQA(BaseDataset, HotPotQAConfig):
    def _create_few_shot_examples(self, examples: BatchType) -> BatchType:
        sentences_grouped = [
            [''.join(sent) for sent in sent_list['sentences']]
            for sent_list in examples['context']
        ]
        sentences_arr = [
            self.sentence_delim.join(sent) for sent in sentences_grouped
        ]
        examples[self.text_col] = [
            self.prefix \
            + sent + self.section_delim \
            + self.suffix \
            + examples['question'][i] + self.section_delim \
            + self.answer_prefix + examples[self.answer_col][i]
            for i, sent in enumerate(sentences_arr)
        ]
        return examples

    def _preprocess(self, examples: BatchType) -> BatchType:
        sentences_grouped = [
            [''.join(sent) for sent in sent_list['sentences']]
            for sent_list in examples['context']
        ]
        sentences_arr = [
            self.sentence_delim.join(sent) for sent in sentences_grouped
        ]
        examples[self.text_col] = [
            self.header + self.section_delim \
            + self.few_shot_examples + self.section_delim \
            + self.prefix \
            + sent + self.section_delim \
            + self.suffix \
            + examples['question'][i] + self.section_delim \
            + self.answer_prefix
            for i, sent in enumerate(sentences_arr)
        ]
        examples = super()._preprocess(examples)
        return examples
