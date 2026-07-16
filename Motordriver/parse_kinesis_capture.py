from __future__ import annotations

import ipaddress
import struct
from pathlib import Path


PCAP = Path(__file__).with_name("captures") / "kinesis_capture.pcapng"
HUB_IP = ipaddress.ip_address("192.168.0.200").packed
PORTS = {40303, 40307, 40308, 40309}


def blocks(data: bytes):
    offset = 0
    while offset + 12 <= len(data):
        block_type, total_length = struct.unpack_from("<II", data, offset)
        if total_length < 12 or offset + total_length > len(data):
            break
        yield block_type, data[offset : offset + total_length]
        offset += total_length


def packet_payloads(pcapng: bytes):
    for block_type, block in blocks(pcapng):
        if block_type != 0x00000006:  # Enhanced Packet Block
            continue
        if len(block) < 32:
            continue
        captured_len = struct.unpack_from("<I", block, 20)[0]
        packet = block[28 : 28 + captured_len]
        if len(packet) < 54:
            continue

        eth_type = struct.unpack_from("!H", packet, 12)[0]
        if eth_type != 0x0800:
            continue

        ip_start = 14
        version_ihl = packet[ip_start]
        ihl = (version_ihl & 0x0F) * 4
        protocol = packet[ip_start + 9]
        if protocol != 6:
            continue

        src_ip = packet[ip_start + 12 : ip_start + 16]
        dst_ip = packet[ip_start + 16 : ip_start + 20]
        if HUB_IP not in (src_ip, dst_ip):
            continue

        tcp_start = ip_start + ihl
        if len(packet) < tcp_start + 20:
            continue
        src_port, dst_port = struct.unpack_from("!HH", packet, tcp_start)
        if src_port not in PORTS and dst_port not in PORTS:
            continue

        tcp_data_offset = (packet[tcp_start + 12] >> 4) * 4
        payload = packet[tcp_start + tcp_data_offset :]
        if not payload:
            continue

        direction = "hub->pc" if src_ip == HUB_IP else "pc->hub"
        hub_port = src_port if src_ip == HUB_IP else dst_port
        yield hub_port, direction, payload


def main() -> None:
    if not PCAP.exists():
        raise SystemExit(f"Missing capture file: {PCAP}")

    for index, (port, direction, payload) in enumerate(packet_payloads(PCAP.read_bytes()), 1):
        preview = payload[:96].hex(" ")
        ascii_preview = "".join(chr(b) if 32 <= b < 127 else "." for b in payload[:96])
        print(f"{index:04d} port={port} {direction} len={len(payload)}")
        print(f"     {preview}")
        print(f"     {ascii_preview}")


if __name__ == "__main__":
    main()
