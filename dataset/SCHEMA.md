# dataset/candidates.jsonl schema (v1)

One row per measured candidate, at any level. Designed so later phases only add rows, never migrate.

| field | type | meaning |
|---|---|---|
| schema | int | schema version (1) |
| level | "pipeline" \| "subgraph" \| "kernel" | granularity of the measurement |
| subgraph_sig | str \| null | model-independent hash of the op subgraph (op types, data-flow edges, dtypes; no names). null at pipeline level |
| model | str | model id the shapes came from (metadata only; never a feature for transfer) |
| hardware_id | str | key into library/hardware/ |
| shapes | object | input shapes / resolution |
| dtype | str | weight / activation precision |
| rewrite | str | rewrite or pattern id applied ("none" = reference) |
| cost_model_pred | object \| null | {time_s, peak_gb} predicted before measuring |
| measured_time_s | float | median measured time |
| measured_peak_gb | float \| null | peak memory |
| error | object | {psnr_db, max_abs, rel_err, lpips} as applicable |
| exp_id | str | link to runs/<exp_id>/ raw logs |
