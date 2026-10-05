from __future__ import annotations

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from .deembed import deembed
from .delivery import build_locate_zip, build_result_zip
from .errors import ComputationError, ParameterError, TouchstoneError
from .reflect import locate_reflections, validate_config
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


def _float_form(value: str | float, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc


def _int_form(value: str | int, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


@app.exception_handler(TouchstoneError)
async def touchstone_error_handler(_request, exc: TouchstoneError):
    detail = {"type": "touchstone_error", "message": str(exc), "file": exc.filename, "line": exc.line}
    return JSONResponse(status_code=400, content={"detail": detail})


@app.exception_handler(ComputationError)
async def computation_error_handler(_request, exc: ComputationError):
    detail = {"type": "computation_error", "message": str(exc), "frequency_hz": exc.frequency_hz}
    return JSONResponse(status_code=422, content={"detail": detail})


@app.exception_handler(ParameterError)
async def parameter_error_handler(_request, exc: ParameterError):
    detail = {"type": "parameter_error", "message": str(exc)}
    return JSONResponse(status_code=400, content={"detail": detail})


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


async def _parse_uploads(measurement, left_fixture, right_fixture):
    uploads = [measurement, left_fixture, right_fixture]
    raw_files = [await upload.read() for upload in uploads]
    names = [
        upload.filename or f"{role}.s2p"
        for role, upload in zip(("measurement", "left_fixture", "right_fixture"), uploads, strict=True)
    ]
    return [
        parse_touchstone(raw, name) for raw, name in zip(raw_files, names, strict=True)
    ]


@app.post("/locate")
async def locate_reflections_endpoint(
    measurement: UploadFile = File(...),
    left_fixture: UploadFile = File(...),
    right_fixture: UploadFile = File(...),
    left_swapped: str | bool = Form(False),
    right_swapped: str | bool = Form(False),
    port: str | int = Form(1),
    velocity_m_per_s: str | float = Form(...),
    distance_min_m: str | float = Form(...),
    distance_max_m: str | float = Form(...),
    amplitude_threshold: str | float = Form(...),
    min_peak_distance_m: str | float = Form(...),
    zero_padding: str | int = Form(1),
) -> Response:
    try:
        swap_left = _bool_form(left_swapped, "left_swapped")
        swap_right = _bool_form(right_swapped, "right_swapped")
        config = validate_config(
            port=_int_form(port, "port"),
            velocity_m_per_s=_float_form(velocity_m_per_s, "velocity_m_per_s"),
            distance_min_m=_float_form(distance_min_m, "distance_min_m"),
            distance_max_m=_float_form(distance_max_m, "distance_max_m"),
            amplitude_threshold=_float_form(amplitude_threshold, "amplitude_threshold"),
            min_peak_distance_m=_float_form(min_peak_distance_m, "min_peak_distance_m"),
            zero_padding=_int_form(zero_padding, "zero_padding"),
        )
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": {"type": "form_error", "message": str(exc)}})

    measured_network, left_network, right_network = await _parse_uploads(measurement, left_fixture, right_fixture)
    device = deembed(
        measured_network,
        left_network,
        right_network,
        left_swapped=swap_left,
        right_swapped=swap_right,
    )
    result = locate_reflections(device, config)
    archive = build_locate_zip(device, result)
    headers = {"Content-Disposition": 'attachment; filename="rfdeembed193-locate.zip"'}
    return Response(content=archive, media_type="application/zip", headers=headers)
