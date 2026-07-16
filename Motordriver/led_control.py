"""Control the tip-LED brightness of the muC112 endoscopes (COMedia C8209HL backend).

The C8209 exposes a vendor UVC extension unit (documented in the "C8209
Controller User Manual", shop.comedia.com.hk). Commands are 32-byte buffers on
control selector 1 of extension-unit node 3:

    byte0 = function ID, byte1.. = parameters
    0x04  LED control     P1: 0=off 1=min 2=medium 3=max
    0x02  White balance   P1: 0=auto 1=fixed
    0x01  Version         SET then GET; reply bytes = BCD year month day

IMPORTANT: the firmware resets the LED to MAX every time streaming starts, so
setting a level while the camera is idle has no lasting effect. Send the
command while the stream is running. Exclusive DirectShow capture (OpenCV
CAP_DSHOW) blocks this script; capture with CAP_MSMF instead - the Windows
frame server then allows this script to run concurrently. The stereo preview
does this automatically via its --left-led/--right-led flags with
--backend msmf. If this script hangs while binding, the camera's UVC stack is
wedged - replug the scope.

Requires: pip install comtypes "pygrabber==0.1"   (Python 3.8 compatible)

Usage:
    python led_control.py --list
    python led_control.py --camera 0 --led medium
    python led_control.py --camera 0 --wb auto
    python led_control.py --camera 0 --version
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from ctypes import POINTER, Structure, byref, c_ubyte, c_ulong, c_void_p, sizeof

import comtypes
from comtypes import COMMETHOD, GUID, HRESULT, IUnknown, COMError

XU_GUID = GUID("{DD880F8A-1CBA-4954-8A25-F7875967F0F7}")
XU_NODE_ID = 3
XU_SELECTOR = 1
XU_DATA_LEN = 32

KSPROPERTY_TYPE_GET = 0x00000001
KSPROPERTY_TYPE_SET = 0x00000002
KSPROPERTY_TYPE_TOPOLOGY = 0x10000000

FUNC_VERSION = 0x01
FUNC_WB = 0x02
FUNC_LED = 0x04

LED_LEVELS = {"off": 0, "min": 1, "medium": 2, "max": 3}
WB_MODES = {"auto": 0, "fixed": 1}


class KSP_NODE(Structure):
    _fields_ = [
        ("Set", GUID),
        ("Id", c_ulong),
        ("Flags", c_ulong),
        ("NodeId", c_ulong),
        ("Reserved", c_ulong),
    ]


class IKsControl(IUnknown):
    _iid_ = GUID("{28F54685-06FD-11D2-B27A-00A0C9223196}")
    _methods_ = [
        COMMETHOD(
            [],
            HRESULT,
            "KsProperty",
            (["in"], POINTER(KSP_NODE), "prop"),
            (["in"], c_ulong, "prop_len"),
            (["in"], c_void_p, "data"),
            (["in"], c_ulong, "data_len"),
            (["out"], POINTER(c_ulong), "bytes_returned"),
        ),
    ]


def list_cameras() -> list:
    from pygrabber.dshow_graph import SystemDeviceEnum, DeviceCategories

    return SystemDeviceEnum().get_available_filters(DeviceCategories.VideoInputDevice)


def open_ks_control(camera_index: int) -> IKsControl:
    from pygrabber.dshow_graph import SystemDeviceEnum, DeviceCategories

    enum = SystemDeviceEnum()
    filt = enum.get_filter_by_index(DeviceCategories.VideoInputDevice, camera_index)
    if isinstance(filt, tuple):
        filt = filt[0]
    return filt.QueryInterface(IKsControl)


def xu_command(ks: IKsControl, payload: bytes, flags: int) -> bytes:
    buf = (c_ubyte * XU_DATA_LEN)(*payload.ljust(XU_DATA_LEN, b"\x00"))
    prop = KSP_NODE()
    prop.Set = XU_GUID
    prop.Id = XU_SELECTOR
    prop.Flags = flags | KSPROPERTY_TYPE_TOPOLOGY
    prop.NodeId = XU_NODE_ID
    prop.Reserved = 0
    ks.KsProperty(byref(prop), sizeof(prop), ctypes.cast(buf, c_void_p), XU_DATA_LEN)
    return bytes(buf)


def set_led(ks: IKsControl, level: int) -> None:
    xu_command(ks, bytes([FUNC_LED, level]), KSPROPERTY_TYPE_SET)


def set_white_balance(ks: IKsControl, mode: int) -> None:
    xu_command(ks, bytes([FUNC_WB, mode]), KSPROPERTY_TYPE_SET)


def read_version(ks: IKsControl) -> str:
    xu_command(ks, bytes([FUNC_VERSION]), KSPROPERTY_TYPE_SET)
    reply = xu_command(ks, bytes([FUNC_VERSION]), KSPROPERTY_TYPE_GET)
    year, month, day = reply[0], reply[1], reply[2]
    return "firmware 20%02x/%02x/%02x" % (year, month, day)


def main() -> None:
    parser = argparse.ArgumentParser(description="muC112 (C8209 backend) tip-LED and WB control.")
    parser.add_argument("--camera", type=int, default=0, help="DirectShow device index (same order as OpenCV DSHOW).")
    parser.add_argument("--led", choices=sorted(LED_LEVELS, key=LED_LEVELS.get), help="Set tip-LED brightness.")
    parser.add_argument("--wb", choices=sorted(WB_MODES, key=WB_MODES.get), help="Set white-balance mode.")
    parser.add_argument("--version", action="store_true", help="Read backend firmware version.")
    parser.add_argument("--list", action="store_true", help="List video devices and exit.")
    args = parser.parse_args()

    comtypes.CoInitialize()

    if args.list:
        for index, name in enumerate(list_cameras()):
            print("%d: %s" % (index, name))
        return
    if not (args.led or args.wb or args.version):
        parser.error("nothing to do: pass --led, --wb, --version or --list")

    try:
        ks = open_ks_control(args.camera)
    except (COMError, ValueError) as exc:
        raise SystemExit(
            "Could not bind camera %d (%s). Close any program using it, or replug the scope."
            % (args.camera, exc)
        )

    if args.led:
        set_led(ks, LED_LEVELS[args.led])
        print("camera %d: LED set to %s" % (args.camera, args.led))
    if args.wb:
        set_white_balance(ks, WB_MODES[args.wb])
        print("camera %d: white balance set to %s" % (args.camera, args.wb))
    if args.version:
        print("camera %d: %s" % (args.camera, read_version(ks)))


if __name__ == "__main__":
    if sys.platform != "win32":
        raise SystemExit("led_control.py uses DirectShow and only runs on Windows.")
    main()
