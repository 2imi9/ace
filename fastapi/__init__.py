"""A lightweight FastAPI stub used for offline testing."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple


class HTTPException(Exception):
    """Exception carrying HTTP metadata."""

    def __init__(self, status_code: int, detail: Any = None) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class BackgroundTasks:
    """Collect and execute deferred callables."""

    def __init__(self) -> None:
        self._tasks: List[Tuple[Callable[..., Any], tuple[Any, ...], Dict[str, Any]]] = []

    def add_task(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self._tasks.append((func, args, kwargs))

    def run(self) -> None:
        for func, args, kwargs in self._tasks:
            func(*args, **kwargs)


class BaseModel:
    """Very small stand-in for pydantic's ``BaseModel``."""

    def __init__(self, **data: Any) -> None:
        for key, value in data.items():
            setattr(self, key, value)

    def dict(self) -> Dict[str, Any]:  # pragma: no cover - compatibility
        return dict(self.__dict__)


class FastAPI:
    """Minimal request router supporting GET and POST."""

    def __init__(self, title: str | None = None, version: str | None = None) -> None:
        self.title = title
        self.version = version
        self._routes: Dict[str, List[Dict[str, Any]]] = {"GET": [], "POST": []}

    # Route registration -------------------------------------------------
    def post(self, path: str, response_model: Any | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._register("POST", path, response_model)

    def get(self, path: str, response_model: Any | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._register("GET", path, response_model)

    def _register(
        self, method: str, path: str, response_model: Any | None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._routes[method].append({"path": path, "func": func, "response_model": response_model})
            return func

        return decorator

    # Introspection helper for the TestClient ----------------------------
    def _find_route(self, method: str, path: str) -> Tuple[Callable[..., Any], Dict[str, str]]:
        routes = self._routes.get(method.upper(), [])
        for route in routes:
            params = _match_path(route["path"], path)
            if params is not None:
                return route["func"], params
        raise KeyError(f"No route for {method} {path}")


def _match_path(template: str, path: str) -> Dict[str, str] | None:
    template_parts = [part for part in template.strip("/").split("/") if part]
    path_parts = [part for part in path.strip("/").split("/") if part]
    if len(template_parts) != len(path_parts):
        return None
    params: Dict[str, str] = {}
    for template_part, path_part in zip(template_parts, path_parts):
        if template_part.startswith("{") and template_part.endswith("}"):
            params[template_part[1:-1]] = path_part
        elif template_part != path_part:
            return None
    return params


__all__ = [
    "BackgroundTasks",
    "BaseModel",
    "FastAPI",
    "HTTPException",
]
