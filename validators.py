from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Optional, TypeVar
import shlex


T = TypeVar("T")


@dataclass
class ValidationResult(Generic[T]):
    value: Optional[T]
    error: Optional[str]

    @property
    def is_valid(self) -> bool:
        return self.error is None


def validate_int(
    value: str | None,
    *,
    default: Optional[int] = None,
    name: str = "value",
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> ValidationResult[int]:
    if value is None or str(value).strip() == "":
        if default is None:
            return ValidationResult(None, f"{name} must be provided")
        return ValidationResult(default, None)
    try:
        parsed = int(str(value).strip())
    except ValueError:
        return ValidationResult(None, f"{name} must be an integer")
    if min_value is not None and parsed < min_value:
        return ValidationResult(None, f"{name} must be >= {min_value}")
    if max_value is not None and parsed > max_value:
        return ValidationResult(None, f"{name} must be <= {max_value}")
    return ValidationResult(parsed, None)


def validate_port(value: str | None, *, default: Optional[int] = None, name: str = "Port") -> ValidationResult[int]:
    return validate_int(value, default=default, name=name, min_value=1, max_value=65535)


def validate_path(value: str, *, must_exist: bool = False, name: str = "Path") -> ValidationResult[Path]:
    try:
        path = Path(value).expanduser().resolve()
    except Exception as exc:
        return ValidationResult(None, f"Invalid {name}: {exc}")
    if must_exist and not path.exists():
        return ValidationResult(None, f"{name} does not exist: {path}")
    return ValidationResult(path, None)


def split_extra_flags(value: str) -> ValidationResult[list[str]]:
    value = (value or "").strip()
    if not value:
        return ValidationResult([], None)
    try:
        return ValidationResult(shlex.split(value), None)
    except ValueError as exc:
        return ValidationResult(None, f"Could not parse extra flags: {exc}")
