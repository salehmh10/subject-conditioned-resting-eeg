# Contributing

Please describe the scientific or usability problem your change solves and identify the affected evidence or source. Documentation corrections, clearer result presentation, and explicit portability improvements are welcome.

Preserve frozen scientific records, saved results, and their hashes. Historical source is retained to explain the completed experiment; changes to experiment behavior should be explicit derivatives with their own version and provenance. Do not silently replace reported values, select favorable seeds, reverse contrast signs without transforming interval bounds, or relabel development evidence as independent confirmation.

Keep public results separate from external data, model weights, waveform caches, local paths/configuration, credentials, and virtual environments. Preserve third-party notices and applicable distribution terms.

For a pull request, explain the change, any effect on scientific interpretation, and what validation was actually performed. Do not infer a fresh test pass from historical records. Documentation-only and packaging work should not trigger training, sampling, bootstrap resampling, or access to Test waveforms.

For future research, use a separate protocol and new output location. The preserved Final Test outcomes must not be reused for tuning. Consult [limitations](docs/LIMITATIONS.md), [reproducibility](docs/REPRODUCIBILITY.md), and [data availability](docs/DATA_AVAILABILITY.md) before proposing execution or artifact distribution.
