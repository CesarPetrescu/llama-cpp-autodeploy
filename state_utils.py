from __future__ import annotations

from dataclasses import MISSING, fields
from typing import Any, Dict


class StrictStateMixin:
    def _field_names(self) -> set[str]:
        return {field.name for field in fields(self)}

    def _field_default(self, key: str) -> Any:
        for field in fields(self):
            if field.name == key:
                if field.default is not MISSING:
                    return field.default
                if field.default_factory is not MISSING:  # type: ignore[comparison-overlap]
                    return field.default_factory()  # type: ignore[misc]
                return None
        return None

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._field_names():
            return getattr(self, key)
        return default

    def setdefault(self, key: str, default: Any) -> Any:
        if key not in self._field_names():
            raise KeyError(key)
        current = getattr(self, key)
        if current is None:
            setattr(self, key, default)
            return default
        return current

    def __getitem__(self, key: str) -> Any:
        if key not in self._field_names():
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self._field_names():
            raise KeyError(key)
        setattr(self, key, value)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in self._field_names()

    def pop(self, key: str, default: Any = None) -> Any:
        if key not in self._field_names():
            return default
        value = getattr(self, key)
        setattr(self, key, self._field_default(key))
        return value

    def update_from_dict(self, data: Dict[str, Any] | None) -> None:
        if not isinstance(data, dict):
            return
        field_names = self._field_names()
        for key, value in data.items():
            if key in field_names:
                setattr(self, key, value)

    def to_dict(self, *, include_private: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for field in fields(self):
            if not include_private and field.name.startswith("_"):
                continue
            payload[field.name] = getattr(self, field.name)
        return payload
