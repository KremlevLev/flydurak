# MaleCNS data

Raw and processed data are ignored by Git. `flysans-prepare all` downloads the MaleCNS v1.0 flat-connectome annotations, predicted neurotransmitters, and neuron-to-neuron connection weights from:

```text
https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
```

The project does not download EM imagery, skeletons, individual synapse coordinates, Steam, or Undertale.

The default build keeps traced neurons and edges with at least three synapses, applies `log1p` to counts, assigns coarse transmitter signs, and normalizes incoming absolute weight. The resulting `.pt` payload records the release, threshold, source URL, body IDs, edge indices, weights, and transmitter groups.

MaleCNS is external data and is not covered by this repository's MIT license. Preserve the dataset's attribution, citation, and license information when publishing derived artifacts. Record checksums and download date for formal reproducibility.
