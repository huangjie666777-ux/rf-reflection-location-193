from __future__ import annotations

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from .deembed import deembed
from .delivery import build_result_zip
from .errors import ComputationError, TouchstoneError
from .touchstone import parse_touchstone

app = FastAPI(
    title="rfdeembed193",
    version="0.1.0",
    description="Two-port Touchstone fixture de-embedding backend.",
)


def _bool_form(value: str | bool, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@app.exception_handler(TouchstoneError)
async def touchstone_error_handler(_request, exc: TouchstoneError):
    detail = {"type": "touchstone_error", "message": str(exc), "file": exc.filename, "line": exc.line}
    return JSONResponse(status_code=400, content={"detail": detail})


@app.exception_handler(ComputationError)
async def computation_error_handler(_request, exc: ComputationError):
    detail = {"type": "computation_error", "message": str(exc), "frequency_hz": exc.frequency_hz}
    return JSONResponse(status_code=422, content={"detail": detail})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "project": "rfdeembed193"}


@app.post("/deembed")
async def deembed_network(
    measurement: UploadFile = File(...),
    left_fixture: UploadFile = File(...),
    right_fixture: UploadFile = File(...),
    left_swapped: str | bool = Form(False),
    right_swapped: str | bool = Form(False),
) -> Response:
    try:
        swap_left = _bool_form(left_swapped, "left_swapped")
        swap_right = _bool_form(right_swapped, "right_swapped")
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": {"type": "form_error", "message": str(exc)}})

    uploads = [measurement, left_fixture, right_fixture]
    raw_files = [await upload.read() for upload in uploads]
    names = [upload.filename or f"{role}.s2p" for role, upload in zip(("measurement", "left_fixture", "right_fixture"), uploads, strict=True)]
    measured_network, left_network, right_network = (
        parse_touchstone(raw, name) for raw, name in zip(raw_files, names, strict=True)
    )
    result = deembed(
        measured_network,
        left_network,
        right_network,
        left_swapped=swap_left,
        right_swapped=swap_right,
    )
    archive = build_result_zip(result)
    headers = {"Content-Disposition": 'attachment; filename="rfdeembed193-result.zip"'}
    return Response(content=archive, media_type="application/zip", headers=headers)
