from typing import Optional, Any, Dict, List
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

@dataclass
class PartialInstantiableConfig:
    _target_: str
    _partial_: bool=True

@dataclass
class InstantiableConfig:
    _target_: str
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplemented

@dataclass
class DatasetConfig(InstantiableConfig):
    name: str = ""
    prefix: str = ""
    suffix: str = " "
    text_col: str = "text"
    label_col: str = "label"
    # If bool then remove all existing columns except label_col
    # If list then remove colummns in the list
    # Real type is Union[str, List[str]]
    remove_columns: Any = True
    batch_size: int = 1000
    batched: bool = True
    load_from_cache: bool = True
    split: Optional[str] = None
    option: Optional[str] = None
    cache_dir: Optional[str] = None

@dataclass
class HotPotQAConfig(DatasetConfig):
    sentence_delim: str = ' '

@dataclass
class MetricConfig(PartialInstantiableConfig):
    output_name: str = "results.json"
    predictions: Dict[str, str] = field(default_factory=dict)
    ground_truths: Dict[str, str] = field(default_factory=dict)

@dataclass
class GenerationAPIConfig:
    """Information about the params here https://beta.openai.com/docs/api-reference/completions/create"""
    n: int = 1
    best_of: int = 1
    logprobs: int = 0
    maxtokens: int = 128
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
class GenerationConfig(InstantiableConfig):
    api: GenerationAPIConfig
    metric: MetricConfig
    dataset: DatasetConfig

cs = ConfigStore.instance()
cs.store(name="base_config", node=GenerationConfig)
cs.store(group="api", name="base_config", node=GenerationAPIConfig)
cs.store(group="metric", name="base_config", node=MetricConfig)
cs.store(group="dataset", name="base_config", node=DatasetConfig)
cs.store(group="dataset", name="hotpot_config", node=HotPotQAConfig)