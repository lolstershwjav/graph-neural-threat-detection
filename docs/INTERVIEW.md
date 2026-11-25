# Technical interview preparation

1. **Why split by capture?** Flows in one capture share hosts, collection patterns and incident timing. Mixing rows can let the model learn capture-specific patterns that persist in the test set.
2. **Why keep unlabeled background flows in the graph?** They describe the observed neighborhood. They provide no supervision and do not enter the confirmed-normal false-positive denominator.
3. **What is the difference between a host label and a flow label?** Flow labels come from the dataset. The host target is derived by checking which nodes touch any positive flow, so both ends can be positive even if only one is implicated.
4. **How does GraphSAGE permit unseen nodes?** The same aggregation weights operate on any node's measurements and neighbors. There is no lookup table keyed by addresses.
5. **Why compare against a node-context tabular model?** A flow-only baseline cannot show whether the benefit came from neighborhood propagation or from basic endpoint activity counts. A context control measures those endpoint signals separately.
6. **How is graph attention normalized?** The model exponentiates each edge logit after subtracting the destination maximum, then divides by the sum over incoming neighbors.
7. **Why remove or rewire edges?** Both keep much of the feature and model pipeline fixed. Removing non-self messages isolates message passing, while shuffling destinations tests whether the genuine adjacency matters.
8. **How is look-ahead avoided?** Flows enter when their recorded durations end. Predictions use only the closed graph window and earlier windows from the same sampled source segment.
9. **What can the dashboard delta establish?** It measures how one trained model's output responds when graph messages are restricted to self edges. It does not identify a root cause or attribute culpability to a host.
10. **What limits the benchmark?** Old single-site data, partial capture segments, uncertain background labels, and a few fixed test captures. Evaluate under independently annotated and more recent traffic before generalizing.

## Evidence-based résumé drafts

- Benchmarked four graph neural network variants against flow and endpoint-context controls across 433,377 CTU-13 flows, using disjoint captures and training-only normalization.
- Measured 97.58% average precision with a graph attention network versus 85.66% for flow-only boosting and used topology ablations to assess how much genuine links contributed.
- Built an offline graph inspector to investigate anonymized host relationships and quantify per-flow prediction sensitivity to graph messages.

The research maps to CrowdStrike's emphasis on testing algorithmic assumptions, designing fair baselines and explaining experimental limits. Practice deriving the destination-wise softmax and explaining the rewiring ablation before using these bullets.
