# Token near-duplicate counts are a false positive unless mean-centred (general warning)

Raw cosine similarity between transformer hidden states is dominated by a direction that every token shares (anisotropy).
Measured raw, many tokens look like near-copies; after subtracting the mean token almost none are.

| Model | Raw near-dup (cos > 0.95) | Mean-centred | Source |
|---|---|---|---|
| FLUX.2 Klein 4B | 53.9 % | 0.016 % | exp-040, exp-041 |
| Wan 2.1 1.3B | 2.4 % | 0.49 % | exp-104 |

The same failure on a second architecture for the same reason, so this is promoted from a Klein finding to a general rule:
**always report the mean-centred figure; never gate a token-merging decision on raw cosine.** On both models token
merging is dropped.
