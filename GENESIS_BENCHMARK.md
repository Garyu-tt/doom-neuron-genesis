# GENESIS / HCP800 — Doom benchmark R01

Goal: replace only the CL1 neural hardware endpoint with the existing HCP800
region-organized digital model, using 200,000 resident simulated neural elements
(250 per parcel across 800 parcels). This is a synthetic region-level model,
not measured single-cell human anatomy. The exact source topology, its license,
and the runtime build must be identified and hashed before any scientific run.
Do not substitute HQ500M's virtual-address count, a random network, or a game bot.

## Reuse

Keep the upstream training server, UDP protocol, scenarios and reward configuration.
The new genesis_neural_interface.py implements the transport only; the actual
GENESIS model factory is an explicit external dependency. No CL SDK or simulator
fallback is used. A real factory must load the original HCP800 graph, construct
250 neural elements in each parcel, preserve the parcel-to-parcel topology and
document the within-parcel and projection construction, neural parameters and
plasticity. A metadata assertion is not independent verification of graph size.

Factory signature:
- factory(topology_path, neurons, seed) -> backend
- metadata() -> dict with architecture="HCP800", regions=800,
  resident_neurons=200000, topology_sha256=<hash of original topology>
- stimulate(frequencies_hz, amplitudes_ua, groups, artifact_wait_ms,
  collect_window_ms) -> eight integer spike counts in upstream group order.
  Document channel-to-population assignment, unit conversion and virtual clock.
  Counts must come from neural state, not the input or a learned policy.
- feedback(kind, channels, frequency_hz, amplitude_ua, pulses, unpredictable,
  event_name) -> apply the recorded feedback to the model; no policy updates.
- event(kind, data) -> log/reset only as required by the experiment.
- close() -> flush telemetry/checkpoint and release resources.

The first controlled comparison uses upstream timing 10 Hz, 50 ms artifact
wait, 50 ms collection and the same scenario, seed and reward configuration.
No privileged information or extra game sensors in the digital arm.
A separate fixed encoder/readout run isolates neural learning.
Use the same eight groups and sampling sizes or explicitly report any mismatch.
Disable HQ500M-style explicit sensory-to-action shortcuts and external Q policy.

## Run when the real factory exists

    python genesis_neural_interface.py --backend YOUR_EXISTING_ADAPTER:factory --topology /path/to/hcp800-topology.json --max-steps 1000
    python training_server.py --mode train --device cpu --cl1-host 127.0.0.1 --max-episodes 1

The adapter name above is a placeholder. This revision is not a runnable
HCP800 implementation on its own. Both processes should run on an isolated
CPU research node, not a production VM. Limit CPU/RAM at the OS level.
UDP is local-only, unordered and does not guarantee delivery. Log packet loss
and timing before comparisons; the bridge does not guarantee hard real time.

## Evidence and acceptance

- Verify 200000 allocated elements, 800 parcels and the original source edge
  count/hash; report modeled synapses separately.
- Record scenario/WAD hash, upstream commit, seed, encoder/decoder checkpoints,
  channel map, tick durations, episode score, steps, spike counts, plasticity,
  elapsed wall time, CPU time and peak RAM.
- Include random spikes, zero spikes, and degree-matched shuffled graph controls;
  match training budget and use several seeds. Compare plasticity on/off.
- First perform a 100-step smoke test before scaling. Save raw logs and reproducible
  configuration with no private paths, secrets, or unlicensed game/data assets.
- Biological comparison remains pending comparable raw CL1 recordings or
  published numerical metrics; video alone is not a quantitative benchmark.

Upstream: https://github.com/SeanCole02/doom-neuron
Related: https://github.com/Garyu-tt/mojum,
https://github.com/Garyu-tt/mikoshi-hq500m (different synthetic architecture).
