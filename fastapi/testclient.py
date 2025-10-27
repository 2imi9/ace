"""Simplified TestClient for the FastAPI stub."""

from __future__ import annotations

import inspect
import json
from typing import Any, Dict, get_type_hints

from . import BackgroundTasks, BaseModel, FastAPI, HTTPException
from .responses import JSONResponse, PlainTextResponse


class Response:
    def __init__(self, *, status_code: int = 200, json_body: Any = None, text: str | None = None) -> None:
        self.status_code = status_code
        self._json = json_body
        if text is not None:
            self.text = text
        elif json_body is not None:
            self.text = json.dumps(json_body)
        else:
            self.text = ""

    def json(self) -> Any:
        if self._json is not None:
            return self._json
        if self.text:
            return json.loads(self.text)
        return None


class TestClient:
    """Invoke FastAPI route handlers directly for testing."""

    __test__ = False

    def __init__(self, app: FastAPI) -> None:
        self.app = app

    def post(self, path: str, *, json: Dict[str, Any] | None = None) -> Response:
        return self._call("POST", path, json=json)

    def get(self, path: str, *, params: Dict[str, Any] | None = None) -> Response:
        return self._call("GET", path, params=params)

    def _call(
        self,
        method: str,
        path: str,
        *,
        json: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
    ) -> Response:
        func, path_params = self.app._find_route(method, path)
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)
        kwargs: Dict[str, Any] = {}
        background_tasks: BackgroundTasks | None = None

        body_consumed = False
        for name, param in signature.parameters.items():
            assigned = False
            if name == "background_tasks":
                background_tasks = BackgroundTasks()
                kwargs[name] = background_tasks
                assigned = True
            elif name in path_params:
                kwargs[name] = path_params[name]
                assigned = True
            elif params and name in params:
                kwargs[name] = params[name]
                assigned = True
            else:
                annotation = type_hints.get(name, param.annotation)
                if (
                    not body_consumed
                    and json is not None
                    and isinstance(annotation, type)
                    and issubclass(annotation, BaseModel)
                ):
                    kwargs[name] = annotation(**json)
                    body_consumed = True
                    assigned = True
            if not assigned:
                if not body_consumed and json is not None:
                    kwargs[name] = json
                    body_consumed = True
                elif param.default is not inspect._empty:  # type: ignore[attr-defined]
                    kwargs[name] = param.default

        try:
            result = func(**kwargs)
        except HTTPException as exc:  # pragma: no cover - defensive
            return Response(status_code=exc.status_code, json_body={"detail": exc.detail})

        if background_tasks is not None:
            background_tasks.run()

        if isinstance(result, JSONResponse):
            return Response(status_code=result.status_code, json_body=result.content)
        if isinstance(result, PlainTextResponse):
            return Response(status_code=result.status_code, text=result.content)
        if isinstance(result, BaseModel):
            return Response(json_body=result.__dict__)
        if isinstance(result, dict):
            return Response(json_body=result)
        if isinstance(result, str):
            return Response(text=result)
        return Response(json_body=result)


__all__ = ["TestClient"]
