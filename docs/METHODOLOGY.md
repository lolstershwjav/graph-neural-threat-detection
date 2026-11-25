# Methodology

## Research questions and hypotheses

Does message passing improve classification of bidirectional network flows when entire captures are held out? Does temporal context provide information beyond current-window host activity? Does the apparent benefit survive rewiring, self-only aggregation, and a tabular control with access to the same node summaries?

The planned primary comparison uses flow average precision. Secondary outcomes include discrimination for an unseen-endpoint slice, incident-host detection, recall and false-positive rate at a fixed validation threshold, and inference time. We hypothesized that true graph edges improve ranking relative to a self-only model; temporal context improves classification where endpoints persist; and graph gains depend on window density and capture identity.

## Input and split

The experiment uses four disjoint byte ranges per public [CTU-13 capture](https://www.stratosphereips.org/datasets-ctu13/), spanning the start, middle, and end. Only the annotated bidirectional flow CSV files are fetched. The source publishes the dataset under [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/). The capture-specific URLs, byte boundaries and SHA-256 hashes are checked in as a manifest. Partial rows are excluded. The reproducibility commands download external data directly. No raw flow files are distributed here.

The split is capture-disjoint. Training uses captures 42, 44, 46, 48, 50, 52 and 54. Validation uses 45, 47 and 51. Testing uses 43, 49 and 53. Capture names are the IDs in the public CTU archive. The split does not hold out every malware family. It evaluates different recording contexts. There is no random mixing of flows within a capture across partitions.

Finish time is estimated as start plus recorded duration. Use 90-second windows over each sampled slice, then score at window close. Deterministically subsample to at most 2,600 completion-ordered flows and deduplicate exact observation rows within each segment. The flow input uses log duration and counters, a source-byte fraction, protocol indicators, and coarse destination-port categories. Each node receives incoming and outgoing degree and mean flow features. Apply training-only normalization. Source and destination addresses build adjacency and evaluate unseen nodes, but do not enter model tensors as IDs or embeddings.

CTU-13 distinguishes verified normal from background traffic. Supervised losses and known-label metrics use only normal and botnet/C&C labels. Background edges contribute graph structure without targets. A positive host target means the host is incident to at least one labeled botnet flow in that window. It may be the counterpart in the observed relationship. It does not identify a compromised device.

## Models and evaluation

- **Flow and context tabular controls.** Histogram gradient boosting on flow features, with or without node summaries at the two endpoints.
- **GCN.** Apply symmetric degree normalization to self-augmented edges.
- **GraphSAGE.** Concatenate a node's representation and its mean neighborhood representation.
- **GAT.** Learn a scalar attention coefficient over each destination's neighborhood, using a stabilized softmax.
- **Temporal GraphSAGE.** Sum a GRU embedding of three prior consecutive closed windows with the current encoding. Histories reset across the disjoint source slices and use explicit missingness.
- **Ablations.** Remove non-self messages, rewire message destinations while retaining the original flow readout, and remove the auxiliary host target. Rewiring preserves endpoint frequencies before duplicate neighbor pairs are collapsed.

Layers use two message-passing steps with 32 hidden units, residual connections and layer normalization. A two-layer edge head combines the ordered endpoint embeddings with local flow measurements. The auxiliary host head uses binary incidence labels. Ignore unlabeled targets in the loss, retain their graph structure, and set positive weights using training labels. Select an epoch by validation flow AP. Set a fixed threshold at the validation benign 95th percentile and measure realized FPR on the test captures.

The unseen-endpoint slice includes any flow with either endpoint absent from all training graphs. Window size cuts use the test median for descriptive slicing after inference. The downloadable evidence reports assessment counts per capture and three deterministic training seeds. Bootstrap intervals summarize seed sensitivity on these fixed captures. No independent test-capture population interval is estimated. Known test labels describe CTU capture annotation. The source does not provide stage-by-stage ground truth for attack chains, so temporal results describe scoring across windows.

## Inspector

Replay anonymized capture-local graph evidence. Host codes are window-local integers. The graph delta removes neighbor messages from the same trained model and keeps the other inputs fixed. It is a perturbation sensitivity check and can be affected by a shift from the training graph. The visualizer exposes the score and the observed label without mapping them to a raw address.

## References

1. García, Grill, Stiborek and Zunino. [An empirical comparison of botnet detection methods](https://doi.org/10.1016/j.cose.2014.05.011). Computers & Security, 2014.
2. Kipf and Welling. [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907). ICLR, 2017.
3. Hamilton, Ying and Leskovec. [Inductive Representation Learning on Large Graphs](https://proceedings.neurips.cc/paper_files/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html). NeurIPS, 2017.
4. Veličković et al. [Graph Attention Networks](https://arxiv.org/abs/1710.10903). ICLR, 2018.
5. Lo et al. [E-GraphSAGE: A Graph Neural Network based Intrusion Detection System for IoT](https://ieeexplore.ieee.org/document/9789878/). IEEE NOMS, 2022.
