"""Minimal Python control for Thorlabs Kinesis KCube DC Servo controllers.

Close or disconnect the Kinesis GUI before using this script; only one process
can normally own the controller connection.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from decimal import Decimal as PyDecimal
from pathlib import Path


KINESIS = Path(r"C:\Program Files\Thorlabs\Kinesis")
DEFAULT_SERIALS = ("27271413", "27271464", "27271523")
DEFAULT_HUB_IP = "192.168.0.200"
DEFAULT_HUB_SERIAL = "120000166"
DEFAULT_HUB_PORTS = (40303, 40307, 40308, 40309)
DISCOVERY_PORT = 40303


def load_kinesis():
    sys.path.append(str(KINESIS))
    import clr  # type: ignore

    clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
    clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")
    clr.AddReference("ThorLabs.MotionControl.KCube.DCServoCLI")

    from System import Decimal  # type: ignore
    from Thorlabs.MotionControl.DeviceManagerCLI import (  # type: ignore
        DeviceConfiguration,
        DeviceManagerCLI,
    )
    from Thorlabs.MotionControl.GenericMotorCLI import MotorDirection  # type: ignore
    from Thorlabs.MotionControl.KCube.DCServoCLI import KCubeDCServo  # type: ignore

    return Decimal, DeviceConfiguration, DeviceManagerCLI, MotorDirection, KCubeDCServo


def net_decimal(value: str):
    Decimal, *_ = load_kinesis()
    return Decimal(PyDecimal(value))


def register_ethernet_endpoints(hub_ip: str, ports: tuple[int, ...]) -> None:
    _, _, DeviceManagerCLI, _, _ = load_kinesis()
    try:
        DeviceManagerCLI.CreateManualDeviceEntry(hub_ip)
    except Exception:
        pass
    for port in ports:
        endpoint = f"{hub_ip}:{port}"
        try:
            DeviceManagerCLI.CreateManualDeviceEntry(endpoint)
        except Exception:
            # Kinesis throws when an entry already exists; that is fine.
            pass


def list_devices(hub_ip: str, ports: tuple[int, ...]) -> None:
    _, _, DeviceManagerCLI, _, _ = load_kinesis()
    register_ethernet_endpoints(hub_ip, ports)
    try:
        for ip_address in DeviceManagerCLI.ScanEthernetRange(hub_ip, hub_ip, DISCOVERY_PORT, 5000):
            DeviceManagerCLI.CreateManualDeviceEntry(ip_address)
            DeviceManagerCLI.CreateManualDeviceEntry(f"{ip_address}:{DISCOVERY_PORT}")
    except Exception:
        pass
    DeviceManagerCLI.BuildDeviceList()
    serials = list(DeviceManagerCLI.GetDeviceList())
    if serials:
        print("\n".join(str(serial) for serial in serials))
    else:
        if kinesis_is_running():
            print("No devices found. The Kinesis GUI is still running; close it, then try again.")
        else:
            print("No devices found. Check that the hub is powered, connected, and configured in Kinesis.")


def kinesis_is_running() -> bool:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-Process Thorlabs.MotionControl.Kinesis -ErrorAction SilentlyContinue",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def connect(serial: str, hub_ip: str = DEFAULT_HUB_IP, ports: tuple[int, ...] = DEFAULT_HUB_PORTS, poll_ms: int = 250):
    _, DeviceConfiguration, DeviceManagerCLI, _, KCubeDCServo = load_kinesis()
    register_ethernet_endpoints(hub_ip, ports)
    DeviceManagerCLI.BuildDeviceList()

    device = KCubeDCServo.CreateKCubeDCServo(serial)
    device.Connect(serial)
    device.StartPolling(poll_ms)
    time.sleep(0.5)
    device.EnableDevice()
    time.sleep(0.5)

    device.LoadMotorConfiguration(
        serial,
        DeviceConfiguration.DeviceSettingsUseOptionType.UseFileSettings,
    )
    return device


def diagnostics(hub_ip: str, ports: tuple[int, ...]) -> None:
    _, _, DeviceManagerCLI, _, _ = load_kinesis()
    register_ethernet_endpoints(hub_ip, ports)
    DeviceManagerCLI.BuildDeviceList()
    print(f"Hub IP: {hub_ip}")
    print(f"Hub serial from Kinesis log: {DEFAULT_HUB_SERIAL}")
    print(f"Manual entries:")
    for entry in DeviceManagerCLI.GetManualDeviceEntries():
        print(f"  {entry.GetComSpec()} | {entry.GetProvider()}")
    print(f"Device types: {list(DeviceManagerCLI.GetDeviceTypesList())}")
    print(f"Devices: {list(DeviceManagerCLI.GetDeviceList())}")
    print("Ethernet scan:")
    for port in ports:
        try:
            found = list(DeviceManagerCLI.ScanEthernetRange(hub_ip, hub_ip, port, 2000))
        except Exception as exc:
            found = [f"{type(exc).__name__}: {exc}"]
        print(f"  {hub_ip}:{port} -> {found}")


def probe(hub_ip: str, ports: tuple[int, ...]) -> None:
    _, _, DeviceManagerCLI, _, KCubeDCServo = load_kinesis()
    import clr  # type: ignore

    clr.AddReference("Thorlabs.MotionControl.KCubeEthernetHubCLI")
    from Thorlabs.MotionControl.KCubeEthernetHubCLI import CubeEthernetHub  # type: ignore

    print(f"Register hub class: {list(CubeEthernetHub.RegisterDevice())}")
    print(f"Register DC servo class: {list(KCubeDCServo.RegisterDevice())}")
    print(f"Scan {hub_ip}:{DISCOVERY_PORT}: {list(DeviceManagerCLI.ScanEthernetRange(hub_ip, hub_ip, DISCOVERY_PORT, 5000))}")
    register_ethernet_endpoints(hub_ip, ports)
    DeviceManagerCLI.BuildDeviceList()
    print(f"Device list after register/scan/build: {list(DeviceManagerCLI.GetDeviceList())}")
    print(f"Hub devices type 120: {list(DeviceManagerCLI.GetDeviceList(120))}")
    print(f"DC servo devices type 27: {list(DeviceManagerCLI.GetDeviceList(27))}")
    for ident in (DEFAULT_HUB_SERIAL, hub_ip, f"{hub_ip}:{DISCOVERY_PORT}"):
        print(f"Hub connect attempt: {ident}")
        hub = CubeEthernetHub.CreateKCubeEthernetHub(ident)
        try:
            hub.CreateConnectionToDevice(ident)
            for _ in range(10):
                if hub.IsConnected:
                    break
                time.sleep(0.5)
            print(f"  state={hub.GetConnectionState()} connected={hub.IsConnected}")
            if hub.IsConnected:
                print(f"  bays={hub.GetBaysCount()}")
        except Exception as exc:
            print(f"  {type(exc).__name__}: {exc}")
        finally:
            try:
                hub.Disconnect(True)
            except Exception:
                pass


def status(serial: str) -> None:
    device = connect(serial)
    try:
        device.RequestPosition()
        time.sleep(0.2)
        print(f"{serial}: position {device.Position} mm")
    finally:
        device.StopPolling()
        device.Disconnect(True)


def home(serial: str, timeout_ms: int) -> None:
    device = connect(serial)
    try:
        print(f"Homing {serial}...")
        device.Home(timeout_ms)
        print(f"{serial}: homed at {device.Position} mm")
    finally:
        device.StopPolling()
        device.Disconnect(True)


def move_to(serial: str, position_mm: str, timeout_ms: int) -> None:
    device = connect(serial)
    try:
        target = net_decimal(position_mm)
        print(f"Moving {serial} to {target} mm...")
        device.MoveTo(target, timeout_ms)
        print(f"{serial}: now at {device.Position} mm")
    finally:
        device.StopPolling()
        device.Disconnect(True)


def move_relative(serial: str, distance_mm: str, timeout_ms: int) -> None:
    _, _, _, MotorDirection, _ = load_kinesis()
    device = connect(serial)
    try:
        distance = PyDecimal(distance_mm)
        direction = MotorDirection.Forward
        if distance < 0:
            direction = MotorDirection.Backward
            distance = -distance
        print(f"Moving {serial} by {distance_mm} mm...")
        device.MoveRelative(direction, net_decimal(str(distance)), timeout_ms)
        print(f"{serial}: now at {device.Position} mm")
    finally:
        device.StopPolling()
        device.Disconnect(True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--serial",
        default=DEFAULT_SERIALS[0],
        help=f"KCube serial number. Seen in Kinesis: {', '.join(DEFAULT_SERIALS)}",
    )
    parser.add_argument("--hub-ip", default=DEFAULT_HUB_IP)
    parser.add_argument(
        "--hub-ports",
        default=",".join(str(port) for port in DEFAULT_HUB_PORTS),
        help="Comma-separated Ethernet hub ports shown in Kinesis.",
    )
    parser.add_argument("--timeout-ms", type=int, default=60000)

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    sub.add_parser("diagnostics")
    sub.add_parser("probe")
    sub.add_parser("status")
    sub.add_parser("home")
    absolute = sub.add_parser("move-to")
    absolute.add_argument("position_mm")
    relative = sub.add_parser("move-by")
    relative.add_argument("distance_mm")

    args = parser.parse_args()
    hub_ports = tuple(int(port.strip()) for port in args.hub_ports.split(",") if port.strip())
    if args.command == "list":
        list_devices(args.hub_ip, hub_ports)
    elif args.command == "diagnostics":
        diagnostics(args.hub_ip, hub_ports)
    elif args.command == "probe":
        probe(args.hub_ip, hub_ports)
    elif args.command == "status":
        status(args.serial)
    elif args.command == "home":
        home(args.serial, args.timeout_ms)
    elif args.command == "move-to":
        move_to(args.serial, args.position_mm, args.timeout_ms)
    elif args.command == "move-by":
        move_relative(args.serial, args.distance_mm, args.timeout_ms)


if __name__ == "__main__":
    main()
