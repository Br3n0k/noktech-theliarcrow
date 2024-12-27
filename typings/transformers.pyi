from typing import Any, Dict, Optional
from torch.nn import Module

class PreTrainedTokenizer:
    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]: ...
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any) -> "PreTrainedTokenizer": ...

class PreTrainedModel(Module):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any) -> "PreTrainedModel": ...
    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None: ...

class LlamaForCausalLM(PreTrainedModel): ...
class LlamaTokenizer(PreTrainedTokenizer): ...

class AutoModelForCausalLM(PreTrainedModel): ...

class TrainingArguments:
    def __init__(
        self,
        output_dir: str,
        num_train_epochs: float,
        per_device_train_batch_size: int,
        save_steps: int,
        save_total_limit: int,
        logging_steps: int,
        fp16: bool,
        **kwargs: Any
    ) -> None: ...

class DataCollatorForLanguageModeling:
    def __init__(self, tokenizer: PreTrainedTokenizer, mlm: bool = True, **kwargs: Any) -> None: ...

class Trainer:
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset: Any,
        data_collator: Any,
        **kwargs: Any
    ) -> None: ...
    def train(self, *args: Any, **kwargs: Any) -> Any: ...

class AutoTokenizer(PreTrainedTokenizer): ...

class TrainerCallback: ...
class TrainerState:
    is_local_process_zero: bool
    epoch: Optional[float]
    global_step: Optional[int]
    learning_rate: Optional[float]
class TrainerControl: ... 