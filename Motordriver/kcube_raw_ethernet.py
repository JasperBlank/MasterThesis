"""Command-line wrapper for raw Ethernet KCube motion control."""

from __future__ import annotations

import argparse

from kcube_motion import (
    CONFIG_PATH,
    COUNTS_PER_MM,
    ENDPOINTS,
    HUB_IP,
    active_hub_sessions,
    get_status,
    hardware_info,
    identify,
    load_axes,
    move_by_counts,
    move_by_mm,
    move_axes_by_mm,
    move_axes_to_mm,
    move_to_counts,
    move_to_mm,
    parse_axis_values,
    read_position_mm,
    recover,
    require_within_limits,
    resolve_target,
    stop,
)


def print_status(serial: str) -> None:
    item = get_status(serial)
    print(f"{serial} on {HUB_IP}:{ENDPOINTS[serial]}")
    if item.position_counts is not None:
        print(f"  position counts: {item.position_counts}")
    if item.dc_status_position_counts is not None:
        print(f"  dc status position counts: {item.dc_status_position_counts}")
        print(f"  dc status position mm: {item.dc_status_position_mm:.6f}")
    if item.status_bits is not None:
        print(f"  status bits: 0x{item.status_bits:08x}")


def print_final_position(final: tuple[int | None, float | None]) -> None:
    counts, mm = final
    if counts is None or mm is None:
        print("  final position unavailable")
        return
    print(f"  final position counts: {counts}")
    print(f"  final position mm: {mm:.6f}")


def main() -> None:
    axes = load_axes()
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", choices=sorted(ENDPOINTS))
    parser.add_argument("--axis", choices=sorted(axes))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("axes")
    sub.add_parser("hw")
    sub.add_parser("status")
    sub.add_parser("status-all")
    sub.add_parser("identify")
    sub.add_parser("list")
    sub.add_parser("ports")
    sub.add_parser("recover")
    sub.add_parser("zero")
    sub.add_parser("zero-all")
    sub.add_parser("home-position")
    sub.add_parser("home-position-all")
    by_counts = sub.add_parser("move-by-counts")
    by_counts.add_argument("counts", type=int)
    to_counts = sub.add_parser("move-to-counts")
    to_counts.add_argument("counts", type=int)
    by_mm = sub.add_parser("move-by-mm")
    by_mm.add_argument("mm", type=float)
    to_mm = sub.add_parser("move-to-mm")
    to_mm.add_argument("mm", type=float)
    move_by = sub.add_parser("move-by")
    move_by.add_argument("axis_offsets", nargs="+", help="Relative moves as axis=mm, for example axis1=0.1 axis2=-0.1")
    move_to = sub.add_parser("move-to")
    move_to.add_argument("axis_targets", nargs="+", help="Absolute targets as axis=mm, for example axis1=1.0 axis2=2.0")
    sub.add_parser("stop")
    args = parser.parse_args()

    if args.serial and args.axis:
        raise SystemExit("Use either --serial or --axis, not both.")
    try:
        serial, axis = resolve_target(args.serial, args.axis, axes)

        if args.command == "list":
            for item_serial, port in ENDPOINTS.items():
                try:
                    info = hardware_info(item_serial)
                    print(f"{item_serial} -> {HUB_IP}:{port} {info.model} {info.description}")
                except Exception as exc:
                    print(f"{item_serial} -> {HUB_IP}:{port} ERROR {exc}")
        elif args.command == "axes":
            if not axes:
                print(f"No axes configured in {CONFIG_PATH}")
                return
            for item in axes.values():
                print(f"{item.name}: serial={item.serial} limits={item.min_mm:g}..{item.max_mm:g} mm")
        elif args.command == "hw":
            info = hardware_info(serial)
            print(f"serial: {info.serial}")
            print(f"model: {info.model}")
            print(f"description: {info.description}")
        elif args.command == "status":
            print_status(serial)
        elif args.command == "status-all":
            for item in axes.values():
                print(f"{item.name}:")
                print_status(item.serial)
        elif args.command == "identify":
            identify(serial)
            print(f"Sent identify to {serial} on {HUB_IP}:{ENDPOINTS[serial]}")
        elif args.command == "move-by-counts":
            current_mm = read_position_mm(serial)
            if current_mm is None:
                raise RuntimeError(f"Could not read current position for {serial}; refusing relative move.")
            require_within_limits(axis, current_mm + args.counts / COUNTS_PER_MM)
            print(f"Sent relative move {args.counts} counts ({args.counts / COUNTS_PER_MM:.6f} mm) to {serial}")
            print_final_position(move_by_counts(serial, args.counts))
        elif args.command == "move-to-counts":
            require_within_limits(axis, args.counts / COUNTS_PER_MM)
            print(f"Sent absolute move to {args.counts} counts ({args.counts / COUNTS_PER_MM:.6f} mm) to {serial}")
            print_final_position(move_to_counts(serial, args.counts))
        elif args.command == "move-by-mm":
            counts = int(round(args.mm * COUNTS_PER_MM))
            print(f"Sent relative move {counts} counts ({counts / COUNTS_PER_MM:.6f} mm) to {serial}")
            print_final_position(move_by_mm(serial, args.mm, axis))
        elif args.command == "move-to-mm":
            counts = int(round(args.mm * COUNTS_PER_MM))
            print(f"Sent absolute move to {counts} counts ({counts / COUNTS_PER_MM:.6f} mm) to {serial}")
            print_final_position(move_to_mm(serial, args.mm, axis))
        elif args.command == "move-by":
            offsets = parse_axis_values(args.axis_offsets, axes)
            for name, offset_mm in offsets.items():
                counts = int(round(offset_mm * COUNTS_PER_MM))
                print(f"{name}: sending relative move {counts} counts ({counts / COUNTS_PER_MM:.6f} mm)")
            results = move_axes_by_mm(offsets, axes)
            for name, final in results.items():
                print(f"{name}:")
                print_final_position(final)
        elif args.command == "move-to":
            targets = parse_axis_values(args.axis_targets, axes)
            for name, target_mm in targets.items():
                counts = int(round(target_mm * COUNTS_PER_MM))
                print(f"{name}: sending absolute move to {counts} counts ({counts / COUNTS_PER_MM:.6f} mm)")
            results = move_axes_to_mm(targets, axes)
            for name, final in results.items():
                print(f"{name}:")
                print_final_position(final)
        elif args.command in ("zero", "home-position"):
            print_final_position(move_to_mm(serial, 0.0, axis))
        elif args.command in ("zero-all", "home-position-all"):
            for item in axes.values():
                print(f"{item.name}:")
                print_final_position(move_to_mm(item.serial, 0.0, item))
        elif args.command == "stop":
            stop(serial)
            print(f"Sent stop to {serial}")
        elif args.command == "ports":
            lines, process_lines = active_hub_sessions()
            if not lines:
                print(f"No active TCP sessions to {HUB_IP}")
                return
            print(f"TCP sessions to {HUB_IP}:")
            for line in lines:
                print(line)
            if process_lines:
                print("\n".join(process_lines))
        elif args.command == "recover":
            print("Checking hub/control port...")
            try:
                hub = hardware_info("120000166")
                print(f"  hub ok: {hub.serial} {hub.model}")
            except Exception as exc:
                print(f"  hub check failed: {exc}")
            for item_serial, result in recover(None).items():
                if isinstance(result, Exception):
                    print(f"  {item_serial}: still unavailable; {result}")
                else:
                    print(f"  {item_serial}: ok {result.serial} {result.model} {result.description}")
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
