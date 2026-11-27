"""Local CL1-compatible transport for an externally supplied GENESIS backend.

No CL SDK, learned decoder, fake neural model, or automatic training.
Backend factory must provide metadata(), stimulate(), feedback(), event(), close().
See GENESIS_BENCHMARK.md for the contract.
"""
import argparse
import importlib
import math
import selectors
import socket
import numpy as np
import udp_protocol as wire

GROUPS = (
    ("encoding", (8, 9, 10, 17, 18, 25, 27, 28)),
    ("move_forward", (41, 42, 49)),
    ("move_backward", (50, 51, 58)),
    ("move_left", (13, 14, 21)),
    ("move_right", (45, 46, 53)),
    ("turn_left", (29, 30, 31, 37)),
    ("turn_right", (59, 60, 61, 62)),
    ("attack", (32, 33, 34)),
)


def validate_metadata(meta):
    if (meta.get("architecture") != "HQ200K"
            or type(meta.get("resident_neurons")) is not int
            or meta["resident_neurons"] != 200_000
            or meta.get("regions") != 7
            or not meta.get("topology_sha256")):
        raise ValueError("Require HQ200K, 7 regions, 200000 resident neurons and topology hash")


def handle_stim(backend, packet, artifact_ms, collect_ms):
    _, freq, amp = wire.unpack_stimulation_command(packet)
    if not np.isfinite(freq).all() or not np.isfinite(amp).all() or (freq < 0).any():
        raise ValueError("Invalid stimulation values")
    counts = np.asarray(backend.stimulate(
        frequencies_hz=freq.tolist(), amplitudes_ua=amp.tolist(),
        groups=GROUPS, artifact_wait_ms=artifact_ms, collect_window_ms=collect_ms
    ), dtype=np.float32)
    if (counts.shape != (8,) or not np.isfinite(counts).all()
            or (counts < 0).any() or (counts != np.floor(counts)).any()):
        raise ValueError("Backend must return eight nonnegative integral spike counts")
    return wire.pack_spike_data(counts)


def serve(backend, args):
    validate_metadata(backend.metadata())
    steps = 0
    with selectors.DefaultSelector() as selector, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as out:
        sockets = []
        try:
            for kind, port in (("feedback", args.feedback_port), ("event", args.event_port),
                               ("stim", args.stim_port)):
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sockets.append(sock)
                sock.bind(("127.0.0.1", port))
                selector.register(sock, selectors.EVENT_READ, kind)
            print("GENESIS transport ready (loopback); metadata:", backend.metadata(), flush=True)
            while steps < args.max_steps:
                ready = selector.select(timeout=args.idle_timeout)
                if not ready:
                    raise TimeoutError("No trainer traffic; stopping bounded run")
                for key, _ in sorted(ready, key=lambda item: {"feedback": 0, "event": 1, "stim": 2}[item[0].data]):
                    packet, _ = key.fileobj.recvfrom(65535)
                    if key.data == "feedback":
                        _, kind, channels, freq, amp, pulses, unpredictable, event_name = wire.unpack_feedback_command(packet)
                        if not math.isfinite(amp) or any(c not in range(64) for c in channels):
                            raise ValueError("Invalid feedback")
                        backend.feedback(kind=kind, channels=channels, frequency_hz=freq,
                                         amplitude_ua=amp, pulses=pulses,
                                         unpredictable=unpredictable, event_name=event_name)
                    elif key.data == "event":
                        _, kind, data = wire.unpack_event_metadata(packet)
                        backend.event(kind, data)
                        if kind == "training_complete":
                            return steps
                    else:
                        response = handle_stim(backend, packet, args.artifact_wait_ms, args.collect_window_ms)
                        out.sendto(response, ("127.0.0.1", args.spike_port))
                        steps += 1
                        if steps >= args.max_steps:
                            return steps
        finally:
            for sock in sockets:
                sock.close()
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, help="Trusted installed module:factory; no fallback")
    parser.add_argument("--topology", required=True, help="Existing HQ200K topology path")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--idle-timeout", type=float, default=60)
    parser.add_argument("--artifact-wait-ms", type=float, default=50)
    parser.add_argument("--collect-window-ms", type=float, default=50)
    for name, port in (("stim", 12345), ("spike", 12346), ("event", 12347), ("feedback", 12348)):
        parser.add_argument("--" + name + "-port", type=int, default=port)
    args = parser.parse_args()
    if (args.max_steps < 1 or not math.isfinite(args.idle_timeout) or args.idle_timeout <= 0
            or not math.isfinite(args.artifact_wait_ms) or args.artifact_wait_ms < 0
            or not math.isfinite(args.collect_window_ms) or args.collect_window_ms <= 0):
        parser.error("Invalid time or step limit")
    module, sep, name = args.backend.partition(":")
    if not sep or not module or not name:
        parser.error("--backend must be module:factory")
    factory = getattr(importlib.import_module(module), name)
    backend = factory(topology_path=args.topology, neurons=200_000, seed=args.seed)
    try:
        print("Completed stimulation steps:", serve(backend, args))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
