import json
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class DatasetRow(BaseModel):
    prompt: str
    completion: str


class EvalResult(BaseModel):
    model_name: str
    data_variant: str
    lora_rank: int | None
    string_match_rate: float | None
    logprob_ratio: float | None
    mcq_rate: float | None


def read_jsonl(fname: str | Path) -> list[dict]:
    results = []
    with Path(fname).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def save_jsonl(
    data: list[T | dict],
    fname: str | Path,
    mode: Literal["a", "w"],
) -> None:
    path = Path(fname)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for item in data:
            datum = item.model_dump() if isinstance(item, BaseModel) else item
            f.write(json.dumps(datum) + "\n")


def save_json(data: BaseModel | dict, fname: str | Path) -> None:
    path = Path(fname)
    path.parent.mkdir(parents=True, exist_ok=True)
    json_data = data.model_dump() if isinstance(data, BaseModel) else data
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f)
