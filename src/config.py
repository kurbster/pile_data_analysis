from typing import Optional, Any, Dict, List
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

@dataclass
class PartialInstantiableConfig:
    _target_: str
    _partial_: bool = True

@dataclass
class InstantiableConfig:
    _target_: str
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplemented

@dataclass
class DatasetConfig(InstantiableConfig):
    """This is the dataset config base class.
    The dataset will be loaded with the huggingface method

        load_dataset(self.name, self.option)
    
    The prompt passed to GPT3 will be constructed as follows.
    The few shot examples can be passed or randomly selected from the dataset.

        <self.header><self.section_delim>
        <FEW_SHOT_EXAMPLES><self.section_delim>
        <self.prefix><TEXT_COL><self.section_delim>
        <self.suffix>

    Each few shot example will be constructed like this

        <self.prefix><TEXT_COL><self.section_delim>
        <self.suffix>
    
    The TEXT_COL will be constructed inside of the self._preprocess method.
    The FEW_SHOT_EXAMPLES will be constructed inside of the self._create_few_shot_examples method
    Any new dataset with extra functionality should override this class.
    """
    name: str = ""
    header: str = ""
    prefix: str = ""
    suffix: str = " "
    text_col: str = "text"
    label_col: str = "label"
    section_delim: str = "\n\n"
    # If bool and True then remove all existing columns except label_col
    # If list then remove colummns in the list
    # Real type is Union[str, List[str]]
    remove_columns: Any = True
    n_few_shot: int = 0
    batch_size: int = 1000
    random_seed: int = 42
    select_n_samples: int = 20
    shuffle: bool = False
    batched: bool = True
    load_from_cache: bool = True
    split: Optional[str] = None
    option: Optional[str] = None
    cache_dir: Optional[str] = None
    few_shot_examples: Optional[List[str]] = field(default_factory=list)

@dataclass
class HotPotQAConfig(DatasetConfig):
    answer_prefix: str = "Answer: "
    sentence_delim: str = ' '

@dataclass
class NQOpenConfig(DatasetConfig):
    id_col: str = "label"
    question_col: str = "question"

@dataclass
class MetricConfig(PartialInstantiableConfig):
    predictions: Dict[str, str] = field(default_factory=dict)
    ground_truths: Dict[str, str] = field(default_factory=dict)

@dataclass
class GenerationAPIConfig:
    """Information about the params here https://beta.openai.com/docs/api-reference/completions/create"""
    n: int = 1
    best_of: int = 1
    logprobs: int = 0
    max_tokens: int = 128
    echo: bool = False
    stream: bool = False
    model: str = "text-davinci-002"
    top_p: float = 1.0
    temperature: float = 0.7
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop: List[str] = field(default_factory=list)
    logit_bias: Dict[str, float] = field(default_factory=dict)

@dataclass
class OutputConfig(PartialInstantiableConfig):
    indent: int = 4
    # Set any of the names to an empty string to skip creating that output
    input_output_df_fname: str = "input_output_pairs.json"
    metric_results_out_fname: str = "metrics.json"
    predictions_output_fname: str = "predictions.json"
    input_dataset_output_fname: str = "dataset.json"
    per_question_metrics_out_fname: str = "per_question_metrics.json"
    dataset: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    predictions: Dict[str, str] = field(default_factory=dict)

@dataclass
class GenerationFuncConfig(PartialInstantiableConfig):
    api_cfg: GenerationAPIConfig = GenerationAPIConfig()
    prompts: List[str] = field(default_factory=list)

@dataclass
class GenerationConfig(InstantiableConfig):
    api: GenerationAPIConfig
    metric: MetricConfig
    dataset: DatasetConfig
    output_func: OutputConfig
    generate_func: GenerationFuncConfig

cs = ConfigStore.instance()
cs.store(name="base_config", node=GenerationConfig)
cs.store(group="api", name="base_config", node=GenerationAPIConfig)
cs.store(group="metric", name="base_config", node=MetricConfig)
cs.store(group="dataset", name="base_config", node=DatasetConfig)
cs.store(group="dataset", name="hotpot_config", node=HotPotQAConfig)
cs.store(group="dataset", name="nq_open_config", node=NQOpenConfig)