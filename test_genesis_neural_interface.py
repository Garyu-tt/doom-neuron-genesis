"""Transport unit tests; fake backend is test-only, not a benchmark."""
import unittest
import numpy as np
import udp_protocol as wire
from genesis_neural_interface import handle_stim, validate_metadata, GROUPS


class Probe:
    def stimulate(self, **kwargs):
        self.kwargs = kwargs
        return list(range(8))


class TransportTest(unittest.TestCase):
    def test_roundtrip(self):
        probe = Probe()
        request = wire.pack_stimulation_command(np.arange(8, dtype=np.float32),
                                                 np.ones(8, dtype=np.float32))
        _, counts = wire.unpack_spike_data(handle_stim(probe, request, 50, 50))
        self.assertEqual(counts.tolist(), list(range(8)))
        self.assertEqual(probe.kwargs["groups"], GROUPS)
        self.assertEqual(probe.kwargs["frequencies_hz"], list(range(8)))
        self.assertEqual(probe.kwargs["collect_window_ms"], 50)

    def test_wrong_model(self):
        with self.assertRaises(ValueError):
            validate_metadata({"architecture": "HCP800", "regions": 800,
                               "resident_neurons": 500_000, "topology_sha256": "test"})

    def test_invalid_packet(self):
        with self.assertRaises(ValueError):
            handle_stim(Probe(), b"", 50, 50)

    def test_wrong_spike_shape(self):
        class Bad:
            def stimulate(self, **kwargs):
                return [1]
        request = wire.pack_stimulation_command(np.ones(8, dtype=np.float32),
                                                 np.ones(8, dtype=np.float32))
        with self.assertRaises(ValueError):
            handle_stim(Bad(), request, 50, 50)


if __name__ == "__main__":
    unittest.main()
