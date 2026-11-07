# Configurations

The model configuration of every run the paper reports for the released designs, as trained. Each JSON is read by
both implementations (`looped_transformer.LoopedTransformerLM.from_json` and the trainer's `LoopedTransformerConfig`).

File names: `<width>_<arm>_<topology>_4klen.json`, topology `pPkKlRcC` = [P → K×R → C] (P prelude blocks, K shared
blocks applied R times, C coda blocks); `dense<N>` = N unique blocks (the standard Transformer).

| arm | `loop_residual_mode` | other fields | design |
|---|---|---|---|
| `dense<N>` | `null` | `num_middle_param_layers` = N − P − C | **Standard**: the depth-matched standard Transformer |
| `plain_*` | `null` | | **Plain Loop**: shared blocks, residual addition |
| `ours_*` | `"delta_attnres"` | `skip0_beta` 0.75 (λ = 1 − β = 1/4) | **Loopback** |
| `bd_<reader>_*` | the reader | `linattn_source` `"delta"` | a reader over the whole-iteration updates (one head) |
| `la_<reader>_*` | the reader | `linattn_source` `"state"`, `attnres_num_heads` 8 | the same reader over the stored states (8 heads) |

Readers: `linattn`, `retnet`, `mamba2`, `gla`, `hgrn2`, `deltanet`, `gdn`, `kda`, `rwkv7`, `lact`, `mesa`.
The paper's six other looped designs (mHC, HC, iHC, Hyperloop, MHAR, AttnRes) are not part of this release.

## Where each result comes from

| paper | directory | files |
|---|---|---|
| 191M, 20 layers: main table, readers, seeds, extra iterations, math/code | `191m_d768x20/` | `dense20`, `plain_p2k4l4c2`, `ours_p2k4l4c2`, `bd_*`/`la_*` at `p2k4l4c2`; `dense8` (the 8-block Standard with Plain Loop's stored parameters) |
| 374M, 24 layers: readers' references, seeds, compute table, Loopback's ablation reference | `374m_d1024x24/` | `dense24`, `plain_p2k10l2c2`, `ours_p2k10l2c2`, `bd_*`/`la_*` at `p2k10l2c2` |
| 374M, 32 layers: main table, extra iterations, math/code | `depth_sweep/d896x32/` | `dense32`, `plain_p2k14l2c2`, `ours_p2k14l2c2` |
| 778M, 24 layers: training-loss figure, compute table | `778m_d1536x24/` | `dense24`, `plain_p2k8l2c6`, `ours_p2k8l2c6` |
| 778M, 48 layers: main table | `depth_sweep/d1088x48/` | `dense48`, `plain_p3k21l2c3`, `ours_p3k21l2c3` |
| 1.4B, 25 layers | `1.4b_d2048x25/` | `dense25`, `plain_p3k8l2c6`, `ours_p3k8l2c6` |
| 3B, 24 layers | `3b_d3072x24/` | `dense24`, `ours_p6k6l2c6` |
| depth sweep (374M and 778M at 16/24/32/48/64 layers) | `depth_sweep/d<width>x<layers>/` | `dense<L>`, `plain_*`, `ours_*` per width |

The depth sweep's 24-layer points are the per-scale 24-layer configurations (identical files kept in both places so that
each directory is complete). Training settings per scale (steps, tokens per step, learning rates) are in
`trainer/README.md` and `results/README.md`.
