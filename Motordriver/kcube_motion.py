"""Reusable raw Ethernet control for KCube DC Servo endpoints behind a KEH hub."""

from __future__ import annotations

import json
import socket
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


HUB_IP = "192.168.0.200"
ENDPOINTS = {
    "27271413": 40307,
    "27271464": 40308,
    "27271523": 40309,
    "120000166": 40303,
}
COUNTS_PER_MM = 512 * 67.49016 / 1.0
CONFIG_PATH = Path(__file__).with_name("kcube_axes.json")


@dataclass
class AxisConfig:
    name: str
    serial: str
    min_mm: float
    max_mm: float


@dataclass
class HardwareInfo:
    serial: str
    model: str
    description: str
    raw: bytes


@dataclass
class AxisStatus:
    serial: str
    position_counts: int | None
    position_mm: float | None
    dc_status_position_counts: int | None
    dc_status_position_mm: float | None
    status_bits: int | None


@dataclass
class VelocityParams:
    serial: str
    min_velocity_counts: int
    acceleration_counts: int
    max_velocity_counts: int

    @property
    def max_velocity_mm_s(self) -> float:
        return self.max_velocity_counts / COUNTS_PER_MM


def load_axes() -> dict[str, AxisConfig]:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    axes: dict[str, AxisConfig] = {}
    for name, item in raw.get("axes", {}).items():
        axes[name] = AxisConfig(
            name=name,
            serial=str(item["serial"]),
            min_mm=float(item["min_mm"]),
            max_mm=float(item["max_mm"]),
        )
    return axes


def resolve_target(serial: str | None, axis: str | None, axes: dict[str, AxisConfig]) -> tuple[str, AxisConfig | None]:
    if axis:
        if axis not in axes:
            raise ValueError(f"Unknown axis '{axis}'. Known axes: {', '.join(sorted(axes)) or '(none)'}")
        return axes[axis].serial, axes[axis]
    if serial:
        return serial, next((item for item in axes.values() if item.serial == serial), None)
    return "27271413", next((item for item in axes.values() if item.serial == "27271413"), None)


def require_within_limits(axis: AxisConfig | None, target_mm: float) -> None:
    if axis is None:
        return
    if not axis.min_mm <= target_mm <= axis.max_mm:
        raise ValueError(
            f"Refusing move: {axis.name} target {target_mm:.6f} mm is outside "
            f"soft limits [{axis.min_mm:.6f}, {axis.max_mm:.6f}] mm."
        )


def parse_axis_values(values: list[str], axes: dict[str, AxisConfig]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected axis=value, got '{value}'")
        name, raw_mm = value.split("=", 1)
        if name not in axes:
            raise ValueError(f"Unknown axis '{name}'. Known axes: {', '.join(sorted(axes)) or '(none)'}")
        try:
            parsed[name] = float(raw_mm)
        except ValueError as exc:
            raise ValueError(f"Invalid millimeter value for {name}: '{raw_mm}'") from exc
    return parsed


def validate_multi_absolute(targets: dict[str, float], axes: dict[str, AxisConfig]) -> None:
    for name, target_mm in targets.items():
        require_within_limits(axes[name], target_mm)


def validate_multi_relative(offsets: dict[str, float], axes: dict[str, AxisConfig]) -> dict[str, float]:
    targets: dict[str, float] = {}
    for name, offset_mm in offsets.items():
        current_mm = read_position_mm(axes[name].serial)
        if current_mm is None:
            raise RuntimeError(f"Could not read current position for {name}; refusing multi-axis move.")
        target_mm = current_mm + offset_mm
        require_within_limits(axes[name], target_mm)
        targets[name] = target_mm
    return targets


def apt_short(message_id: int, param1: int = 0, param2: int = 0, dest: int = 0x50, source: int = 0x01) -> bytes:
    return struct.pack("<HBBBB", message_id, param1, param2, dest, source)


def apt_long(message_id: int, payload: bytes, dest: int = 0x50, source: int = 0x01) -> bytes:
    return struct.pack("<HHBB", message_id, len(payload), dest | 0x80, source) + payload


def recv_available(sock: socket.socket, timeout: float = 0.35) -> bytes:
    sock.settimeout(timeout)
    chunks: list[bytes] = []
    while True:
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            break
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def send_recv(port: int, *messages: bytes, timeout: float = 0.35, connect_timeout: float = 2.0) -> bytes:
    try:
        with socket.create_connection((HUB_IP, port), timeout=connect_timeout) as sock:
            for message in messages:
                sock.sendall(message)
                time.sleep(0.03)
            return recv_available(sock, timeout=timeout)
    except OSError as exc:
        raise RuntimeError(
            f"Could not connect to {HUB_IP}:{port}. Close Kinesis and, if only TIME_WAIT remains, wait or power-cycle the hub."
        ) from exc


def parse_messages(data: bytes) -> list[tuple[int, bytes]]:
    messages: list[tuple[int, bytes]] = []
    i = 0
    while i + 6 <= len(data):
        if data[i : i + 6] == b"\x00" * 6:
            i += 6
            continue
        msg_id, length, dest, source = struct.unpack_from("<HHBB", data, i)
        if dest & 0x80:
            total = 6 + length
            if i + total > len(data):
                break
            messages.append((msg_id, data[i + 6 : i + total]))
            i += total
        else:
            messages.append((msg_id, data[i : i + 6]))
            i += 6
    return messages


def hardware_info(serial: str) -> HardwareInfo:
    port = ENDPOINTS[serial]
    data = send_recv(port, apt_short(0x0005), timeout=0.6)
    for msg_id, payload in parse_messages(data):
        if msg_id == 0x0006 and len(payload) >= 84:
            serial_no = str(struct.unpack_from("<I", payload, 0)[0])
            model = payload[4:12].split(b"\x00", 1)[0].decode("ascii", errors="ignore")
            description = payload[18:66].split(b"\x00", 1)[0].decode("ascii", errors="ignore")
            return HardwareInfo(serial_no, model, description, payload)
    raise RuntimeError(f"No hardware info reply from {serial} on port {port}. Raw: {data.hex(' ')}")


def get_status(serial: str) -> AxisStatus:
    port = ENDPOINTS[serial]
    data = send_recv(port, apt_short(0x0411, 1), apt_short(0x0490, 1), timeout=0.6)
    position_counts = None
    dc_position_counts = None
    status_bits = None
    for msg_id, payload in parse_messages(data):
        if msg_id == 0x0412 and len(payload) >= 6:
            position_counts = struct.unpack_from("<i", payload, 2)[0]
        elif msg_id == 0x0491 and len(payload) >= 14:
            dc_position_counts = struct.unpack_from("<i", payload, 2)[0]
            status_bits = struct.unpack_from("<I", payload, len(payload) - 4)[0]
    return AxisStatus(
        serial=serial,
        position_counts=position_counts,
        position_mm=None if position_counts is None else position_counts / COUNTS_PER_MM,
        dc_status_position_counts=dc_position_counts,
        dc_status_position_mm=None if dc_position_counts is None else dc_position_counts / COUNTS_PER_MM,
        status_bits=status_bits,
    )


def identify(serial: str) -> None:
    send_recv(ENDPOINTS[serial], apt_short(0x0223), timeout=0.2)


def get_position_counts(serial: str) -> int | None:
    data = send_recv(ENDPOINTS[serial], apt_short(0x0490, 1), timeout=0.5)
    for msg_id, payload in parse_messages(data):
        if msg_id == 0x0491 and len(payload) >= 6:
            return struct.unpack_from("<i", payload, 2)[0]
    return None


def read_position_mm(serial: str) -> float | None:
    pos = get_position_counts(serial)
    if pos is None:
        return None
    return pos / COUNTS_PER_MM


def wait_for_position(serial: str, timeout_s: float = 30.0) -> tuple[int | None, float | None]:
    start = time.monotonic()
    last_pos = None
    stable_reads = 0
    while time.monotonic() - start < timeout_s:
        pos = get_position_counts(serial)
        if pos is not None:
            if pos == last_pos:
                stable_reads += 1
            else:
                stable_reads = 0
                last_pos = pos
            if stable_reads >= 3:
                return pos, pos / COUNTS_PER_MM
        time.sleep(0.25)
    return last_pos, None if last_pos is None else last_pos / COUNTS_PER_MM


def move_by_counts(serial: str, counts: int, wait: bool = True) -> tuple[int | None, float | None]:
    payload = struct.pack("<Hi", 1, counts)
    send_recv(ENDPOINTS[serial], apt_long(0x0448, payload), timeout=0.2)
    if wait:
        return wait_for_position(serial)
    return None, None


def move_to_counts(serial: str, counts: int, wait: bool = True) -> tuple[int | None, float | None]:
    payload = struct.pack("<Hi", 1, counts)
    send_recv(ENDPOINTS[serial], apt_long(0x0453, payload), timeout=0.2)
    if wait:
        return wait_for_position(serial)
    return None, None


def get_velocity_params(serial: str) -> VelocityParams:
    data = send_recv(ENDPOINTS[serial], apt_short(0x0414, 1), timeout=0.5)
    for msg_id, payload in parse_messages(data):
        if msg_id == 0x0415 and len(payload) >= 14:
            return VelocityParams(
                serial=serial,
                min_velocity_counts=struct.unpack_from("<i", payload, 2)[0],
                acceleration_counts=struct.unpack_from("<i", payload, 6)[0],
                max_velocity_counts=struct.unpack_from("<i", payload, 10)[0],
            )
    raise RuntimeError(f"No velocity parameter reply from {serial}. Raw: {data.hex(' ')}")


def set_velocity_params(serial: str, params: VelocityParams) -> None:
    payload = struct.pack(
        "<Hiii",
        1,
        int(params.min_velocity_counts),
        int(params.acceleration_counts),
        int(params.max_velocity_counts),
    )
    send_recv(ENDPOINTS[serial], apt_long(0x0413, payload), timeout=0.2)


def set_velocity_scale(serial: str, scale: float) -> VelocityParams:
    if scale <= 0:
        raise ValueError("Velocity scale must be positive.")
    original = get_velocity_params(serial)
    scaled = VelocityParams(
        serial=serial,
        min_velocity_counts=original.min_velocity_counts,
        acceleration_counts=original.acceleration_counts,
        max_velocity_counts=max(1, int(round(original.max_velocity_counts * scale))),
    )
    set_velocity_params(serial, scaled)
    return original


def move_velocity(serial: str, direction: int) -> None:
    if direction not in (-1, 1):
        raise ValueError("Velocity direction must be -1 or 1.")
    # For these KCube DC Servo endpoints, param2=2 increases the reported
    # position and param2=1 decreases it.
    param2 = 2 if direction > 0 else 1
    send_recv(ENDPOINTS[serial], apt_short(0x0457, 1, param2), timeout=0.2)


def move_by_mm(serial: str, mm: float, axis: AxisConfig | None, wait: bool = True) -> tuple[int | None, float | None]:
    current_mm = read_position_mm(serial)
    if current_mm is None:
        raise RuntimeError(f"Could not read current position for {serial}; refusing relative move.")
    require_within_limits(axis, current_mm + mm)
    return move_by_counts(serial, int(round(mm * COUNTS_PER_MM)), wait=wait)


def move_to_mm(serial: str, mm: float, axis: AxisConfig | None, wait: bool = True) -> tuple[int | None, float | None]:
    require_within_limits(axis, mm)
    return move_to_counts(serial, int(round(mm * COUNTS_PER_MM)), wait=wait)


def move_axes_by_mm(
    offsets: dict[str, float],
    axes: dict[str, AxisConfig],
    wait: bool = True,
) -> dict[str, tuple[int | None, float | None]]:
    validate_multi_relative(offsets, axes)
    for name, offset_mm in offsets.items():
        move_by_counts(axes[name].serial, int(round(offset_mm * COUNTS_PER_MM)), wait=False)
    if not wait:
        return {name: (None, None) for name in offsets}
    return {name: wait_for_position(axes[name].serial) for name in offsets}


def move_axes_to_mm(
    targets: dict[str, float],
    axes: dict[str, AxisConfig],
    wait: bool = True,
) -> dict[str, tuple[int | None, float | None]]:
    validate_multi_absolute(targets, axes)
    for name, target_mm in targets.items():
        move_to_counts(axes[name].serial, int(round(target_mm * COUNTS_PER_MM)), wait=False)
    if not wait:
        return {name: (None, None) for name in targets}
    return {name: wait_for_position(axes[name].serial) for name in targets}


def stop(serial: str) -> None:
    send_recv(ENDPOINTS[serial], apt_short(0x0465, 1, 2), timeout=0.2)


def active_hub_sessions() -> tuple[list[str], list[str]]:
    result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, check=False)
    lines = [line for line in result.stdout.splitlines() if HUB_IP in line]
    pids = sorted({line.split()[-1] for line in lines if line.split()[-1].isdigit() and line.split()[-1] != "0"})
    process_lines: list[str] = []
    if pids:
        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-Process -Id " + ",".join(pids) + " | Select-Object Id,ProcessName | Format-Table -AutoSize",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        process_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return lines, process_lines


def recover(serial: str | None = None) -> dict[str, HardwareInfo | Exception]:
    results: dict[str, HardwareInfo | Exception] = {}
    serials = [serial] if serial else [s for s in ENDPOINTS if s != "120000166"]
    for item in serials:
        for _ in range(7):
            try:
                results[item] = hardware_info(item)
                break
            except Exception as exc:
                results[item] = exc
                time.sleep(2)
    return results
