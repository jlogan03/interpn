from __future__ import annotations

import json
from typing import Union, Annotated, Literal, Any

import numpy as np
from numpy.typing import NDArray

from pydantic import (
    field_validator,
    field_serializer,
    ConfigDict,
    BaseModel,
    Field,
)


class ArrayF64(BaseModel):
    """
    Serializable wrapper for NDArray[float64].
    """
    data: NDArray[np.float64]
    dtype: Literal["float64"] = "float64"

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    @field_validator("data", mode="before")
    def _validate_x(data: Any) -> NDArray[np.float64]:
        if isinstance(data, str):
            y = np.ascontiguousarray(np.array(json.loads(data), dtype=np.float64))
        elif isinstance(data, np.ndarray):
            y = np.ascontiguousarray(data.astype(np.float64))
        elif isinstance(data, list):
            y = np.array(data, dtype=np.float64)
        else:
            raise TypeError

        return y

    @field_serializer("data", return_type=str)
    def _serialize_x(data: Any) -> str:
        return json.dumps(data.tolist())


class ArrayF32(BaseModel):
    """
    Serializable wrapper for NDArray[float32].
    
    The data is represented as a list of float64 on disk and in RAM 
    during serialization and deserialization.
    """
    data: NDArray[np.float32]
    dtype: Literal["float32"] = "float32"

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    @field_validator("data", mode="before")
    def _validate_x(data: Any) -> NDArray[np.float32]:
        if isinstance(data, str):
            y = np.ascontiguousarray(np.array(json.loads(data), dtype=np.float32))
        elif isinstance(data, np.ndarray):
            y = np.ascontiguousarray(data.astype(np.float32))
        elif isinstance(data, list):
            y = np.array(data, dtype=np.float32)
        else:
            raise TypeError

        return y

    @field_serializer("data", return_type=str)
    def _serialize_x(data: Any) -> str:
        return json.dumps(data.tolist())


Array = Annotated[Union[ArrayF32, ArrayF64], Field(discriminator="dtype")]
