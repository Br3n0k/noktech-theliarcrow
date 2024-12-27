from typing import Any, List, Optional, Union, Callable, Sequence, Mapping, TypeVar

T = TypeVar('T')

class Dataset:
    column_names: List[str]
    def map(self, function: Callable[..., T], batched: bool = False, remove_columns: Optional[List[str]] = None, **kwargs: Any) -> "Dataset": ...

class DatasetDict:
    def __getitem__(self, key: str) -> Dataset: ...
    def __contains__(self, key: str) -> bool: ...
    def map(self, function: Callable[..., T], batched: bool = False, remove_columns: Optional[List[str]] = None, **kwargs: Any) -> "DatasetDict": ...

def load_dataset(
    path: str,
    name: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    **kwargs: Any
) -> Union[Dataset, DatasetDict]: ... 