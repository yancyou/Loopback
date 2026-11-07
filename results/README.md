# Results

Every number here is from the paper (arXiv:2611.07226), for the designs this repository releases: the standard
Transformer (**Standard**), the looped Transformer with residual addition (**Plain Loop**), **Loopback**, and the
eleven loop-axis readers. The paper's tables also compare six other looped designs (mHC, HC, iHC, Hyperloop, MHAR,
AttnRes); see the paper for those columns.

Protocols (`scripts/`): validation loss is the held-out ClimbMix shard packed as in training, the first 1,024 windows of
4,096 tokens, bf16, final checkpoint (`eval_climb_val.py`); downstream tasks are lm-evaluation-harness 0.4.12, zero-shot,
full sets, 1,000 bootstrap iterations, every prompt starting with the beginning-of-sequence token, length-normalized
accuracy for HellaSwag, OpenBookQA, ARC-Easy and ARC-Challenge, MMLU as one task in the unweighted thirteen-task mean,
WikiText word perplexity reported separately (`eval_downstream.py`, `aggregate_downstream.py`). One run per arm,
seed 42, unless stated. Transformer parameters exclude the embedding and the vocabulary head; shared blocks count once.

## Training settings

| scale | width | blocks | heads | SwiGLU | Loopback | seq./step | micro × acc. | steps | tokens | peak LR Standard / looped | shards |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 191M | 768 | 20 | 12 | 2,048 | [2 → 4×4 → 2] | 128 | 16 × 1 | 7,280 | 3.82B | 1e-3 / 3e-3 | 0–239 |
| 374M | 1,024 | 24 | 16 | 2,816 | [2 → 10×2 → 2] | 256 | 8 × 4 | 7,120 | 7.47B | 2e-3 / 2e-3 | 0–239 |
| 778M | 1,536 | 24 | 24 | 4,096 | [2 → 8×2 → 6] | 512 | 8 × 8 | 7,440 | 15.60B | 1e-3 / 1.5e-3 | 0–239 |
| 1.4B | 2,048 | 25 | 32 | 5,632 | [3 → 8×2 → 6] | 960 | 8 × 15 | 7,200 | 28.31B | 1e-3 / 1.5e-3 | 0–559 |
| 3B | 3,072 | 24 | 48 | 8,192 | [6 → 6×2 → 6] | 1,920 | 4 × 60 | 7,600 | 59.77B | 1e-3 / 1e-3 | 0–559 |

Shared by every run: AdamW (0.9, 0.95), ε 1e-15, weight decay 0, gradient clip 1.0, sequences of 4,096 tokens on 8 GPUs,
linear warmup 128 steps, peak held, the last 10% of steps decayed to 0 along 1 − √u, initialization N(0, 0.02²), the
Llama 32k tokenizer, ClimbMix. Budgets are 20 tokens per parameter of Standard's total count (the 3B budget, 59.77B
tokens, is 20.5 TPP of its 2,914.67M). The 32- and 48-layer settings and the depth sweep keep each scale's token budget
and learning rates (374M: 2e-3 for every arm; 778M: 1e-3 Standard, 1.5e-3 looped). Loopback uses λ = 1/4 everywhere.

## Main comparison at matched tokens and executed depth

| setting | model | Transformer params (M) | ClimbMix loss ↓ | 13-task mean ↑ | WikiText ppl ↓ |
|---|---|---|---|---|---|
| 191M, 20 layers, [2 → 4×4 → 2] | Standard | 141.59 | 2.6234 | 38.28 | 46.30 |
| | Plain Loop | 56.64 | 2.6802 | 36.64 | 50.30 |
| | **Loopback** | 56.65 | **2.6187** | **38.64** | **45.74** |
| 374M, 32 layers, [2 → 14×2 → 2] | Standard | 323.02 | 2.4395 | 39.18 | 34.79 |
| | Plain Loop | 181.70 | 2.4371 | 40.84 | 34.57 |
| | **Loopback** | 181.72 | **2.4008** | **40.89** | **32.61** |
| 778M, 48 layers, [3 → 21×2 → 3] | Standard | 708.68 | 2.2644 | 43.04 | 26.39 |
| | Plain Loop | 398.63 | 2.2737 | 42.78 | 26.78 |
| | **Loopback** | 398.68 | **2.2404** | **43.58** | **25.33** |
| 1.4B, 25 layers, [3 → 8×2 → 6] | Standard | 1,284.61 | 2.1283 | 46.33 | 21.09 |
| | Plain Loop | 873.54 | 2.1290 | – | – |
| | **Loopback** | 873.57 | **2.1073** | **46.50** | **20.29** |
| 3B, 24 layers, [6 → 6×2 → 6] | Standard | 2,718.06 | 2.0005 | 48.56 | 17.28 |
| | **Loopback** | 2,038.58 | **1.9849** | **49.45** | **16.74** |

At 374M with 24 layers ([2 → 10×2 → 2], the readers' and the compute table's setting): Standard 2.4198, Plain Loop 2.4415,
Loopback 2.3971. The 191M eight-block Standard that matches Plain Loop's stored parameters (56.64M) reaches 2.7376.

Per task at 1.4B and 3B (zero-shot accuracy, %):

| task | 1.4B Standard | 1.4B Loopback | 3B Standard | 3B Loopback |
|---|---|---|---|---|
| PIQA | 73.88 | **74.48** | **77.04** | 75.90 |
| HellaSwag | 55.02 | **56.43** | 62.84 | **63.72** |
| WinoGrande | 56.51 | **56.99** | 57.22 | **58.48** |
| OpenBookQA | **37.80** | 36.60 | 40.40 | **41.80** |
| CommonsenseQA | **23.34** | 18.67 | 19.90 | **21.87** |
| BoolQ | 62.51 | **63.24** | 62.23 | **64.95** |
| RACE | **34.64** | 33.49 | 35.12 | **36.17** |
| ARC-Easy | 66.16 | **67.00** | 69.19 | **71.13** |
| ARC-Challenge | 36.35 | **37.03** | 40.53 | **40.96** |
| MMLU (57 subjects) | 24.70 | **26.81** | 26.26 | **28.86** |
| ANLI-R1 | 31.30 | **34.80** | 31.00 | **32.90** |
| RTE | **57.40** | 55.23 | **61.01** | 56.68 |
| LAMBADA | 42.64 | **43.72** | 48.50 | **49.45** |
| thirteen-task mean | 46.33 | **46.50** | 48.56 | **49.45** |

At 3B the bootstrap standard error of each thirteen-task mean is 0.40 points. The per-task 191M, 374M and 778M numbers
are in the paper's main table.

## Run-to-run variation (validation loss, seeds 42 / 1337)

| | Standard | Plain Loop | Loopback |
|---|---|---|---|
| 191M, 20 layers | 2.6234 / 2.6144 | 2.6802 / 2.6815 | 2.6187 / 2.6155 |
| 374M, 24 layers | 2.4198 / 2.4257 | – | 2.3971 / 2.3996 |

Each seed sets the initialization and the data order. At 191M, Loopback − Standard is −0.0047 / +0.0011 (a third seed
gives −0.0064), so that comparison is within run-to-run variation; at 374M it is −0.0227 / −0.0261.

## Readers (validation loss ↓)

| reader | 191M, updates | 191M, states | 374M (24 layers), updates | 374M, states |
|---|---|---|---|---|
| Gated DeltaNet | **2.6489** | 2.6685 | 2.4308 | 2.4333 |
| HGRN-2 | 2.6522 | 2.6619 | **2.4256** | 2.4333 |
| RWKV-7 | 2.6532 | 2.6791 | 2.4371 | 2.4398 |
| Mesa | 2.6539 | **2.6603** | 2.4277 | 2.4342 |
| DeltaNet | 2.6542 | 2.6664 | 2.4310 | 2.4329 |
| KDA | 2.6543 | 2.6612 | 2.4290 | **2.4295** |
| Mamba-2 | 2.6544 | 2.6759 | 2.4308 | 2.4319 |
| LaCT-inspired | 2.6558 | 2.6774 | 2.4308 | 2.4336 |
| GLA | 2.6640 | 2.6773 | 2.4352 | 2.4419 |
| RetNet | 2.6650 | 2.6827 | 2.4338 | 2.4396 |
| Linear attention | 2.6697 | 2.6846 | 2.4346 | 2.4417 |
| Plain Loop | 2.6802 | 2.6802 | 2.4415 | 2.4415 |
| Standard | 2.6234 | 2.6234 | 2.4198 | 2.4198 |
| **Loopback** | **2.6187** | **2.6187** | **2.3971** | **2.3971** |

Updates = the whole-iteration updates with one head (`bd_*` configs); states = the stored states with eight heads
(`la_*`). Every reader improves on Plain Loop; none reaches Standard or Loopback.

## Depth at fixed parameters (validation loss ↓, two iterations, fixed tokens)

| layers | width | topology | 374M Standard | 374M Plain Loop | 374M Loopback | | width | 778M Standard | 778M Plain Loop | 778M Loopback |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 1,216 | [1 → 7×2 → 1] | 2.4218 | 2.4536 | **2.4191** | | 1,856 | 2.2604 | 2.2857 | **2.2583** |
| 24 | 1,024 | [2 → 10×2 → 2] | 2.4198 | 2.4415 | **2.3971** | | 1,536 | 2.2441 | 2.2713 | **2.2425** |
| 32 | 896 | [2 → 14×2 → 2] | 2.4395 | 2.4371 | **2.4008** | | 1,344 | 2.2471 | 2.2737 | **2.2415** |
| 48 | 768 | [3 → 21×2 → 3] | 2.4544 | 2.4425 | **2.4005** | | 1,088 | 2.2644 | 2.2737 | **2.2404** |
| 64 | 640 | [4 → 28×2 → 4] | 2.4958 | 2.4695 | **2.4224** | | 960 | 2.2788 | 2.2816 | **2.2512** |

Width varies with depth so that Standard's total parameters stay within 5% of its 24-layer reference. The 374M sweep ran
on one node of eight H100s, the 778M sweep (including the 48-layer main-table models) on nodes of eight A100s.

## More iterations at inference, without retraining (validation loss)

The trained checkpoint is run with R′ iterations of its shared stack (`scripts/eval_extra_visits.py`); Loopback's
destinations beyond the trained depth reuse the last trained iteration's queries.

| model | trained R | R′ = 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| 191M Loopback | 4 | – | – | **2.6187** | 2.6390 | 2.6639 | 2.6791 |
| 191M Plain Loop | 4 | – | – | **2.6802** | 2.7341 | 2.8518 | 3.1335 |
| 374M (32 layers) Loopback | 2 | **2.4008** | 2.4275 | 2.4751 | – | – | – |
| 374M (32 layers) Plain Loop | 2 | **2.4371** | 2.5355 | 2.7066 | – | – | – |
| 1.4B Loopback | 2 | **2.1073** | 2.1179 | 2.1354 | – | – | – |
| 1.4B Plain Loop | 2 | **2.1290** | 2.2064 | 2.3770 | – | – | – |

At 3B the paper reports the thirteen-task mean instead: Loopback 49.45% at the trained R = 2 and 50.17% with one extra
iteration (Standard at its fixed depth: 48.56%); the paper's appendix lists the individual tasks.

## Mathematical and code text

Teacher-forced loss of the reference solutions (nats per solution token, ↓; `scripts/eval_math_code.py`) and five-shot
numeric-answer accuracy on six word-problem sets (%).

| | 191M Standard | 191M Plain Loop | 191M Loopback | 374M-32L Standard | 374M-32L Plain Loop | 374M-32L Loopback |
|---|---|---|---|---|---|---|
| GSM8K | 1.822 | 1.865 | **1.768** | 1.600 | 1.612 | **1.529** |
| MATH-500 | **1.967** | 2.028 | 1.972 | 1.755 | 1.755 | **1.707** |
| HumanEval | **1.827** | 1.935 | 1.842 | 1.518 | 1.520 | **1.447** |
| MBPP | 2.021 | 2.118 | **2.018** | 1.690 | 1.707 | **1.643** |
| mean | 1.909 | 1.987 | **1.900** | 1.641 | 1.648 | **1.582** |

| | 1.4B Standard | 1.4B Loopback | 3B Standard | 3B Loopback |
|---|---|---|---|---|
| GSM8K | 1.275 | **1.227** | 1.143 | **1.086** |
| MATH-500 | 1.421 | **1.391** | 1.289 | **1.259** |
| HumanEval | 1.065 | **1.035** | 0.906 | **0.868** |
| MBPP | 1.282 | **1.235** | 1.104 | **1.078** |
| four-set mean | 1.261 | **1.222** | 1.110 | **1.073** |
| word problems (%) ↑ | 2.0 | **2.6** | 2.6 | **6.4** |

Paired bootstrap standard errors of the differences between designs are at most 0.003 on the four-set mean.

## Compute and efficiency (eight H100s for training, one H100 for prefill)

| scale | model | Transformer params (M) | extra forward FLOPs | train tok/s/GPU | peak GiB/GPU | prefill tok/s |
|---|---|---|---|---|---|---|
| 191M, [2 → 4×4 → 2] | Standard | 141.59 | – | 222k | 38 | 885k |
| | Plain Loop | 56.64 | +0.00% | 226k | 38 | 886k |
| | Loopback | 56.65 | +0.08% | 167k | 42 | 601k |
| 374M, [2 → 10×2 → 2] | Standard | 308.33 | – | 128k | 26 | 492k |
| | Plain Loop | 179.86 | +0.00% | 130k | 26 | 491k |
| | Loopback | 179.88 | +0.08% | 98k | 32 | 352k |
| 778M, [2 → 8×2 → 6] | Standard | 679.55 | – | 75k | 38 | 269k |
| | Plain Loop | 453.04 | +0.00% | 76k | 37 | 269k |
| | Loopback | 453.06 | +0.04% | 63k | 44 | 217k |

Loopback trains and prefills with its fused read, in the form the paper measured; the released kernels are faster (see the
README's training section). FLOPs are counted on one 4,096-token sequence; training throughput and
peak memory are the runs' own logs (Loopback: 160-step timing runs on the same node and batch); prefill is a batch of
eight 4,096-token sequences through one H100. At 3B the logged training throughputs are 24.7k tokens/s/GPU for Standard
and 23.3k for Loopback.

Persistent inference memory, analytic (bf16 weights including the embedding and the head, plus a KV cache of
2 × executed depth × width × 2 bytes per token; totals for one and eight 4,096-token sequences):

| scale | model | total params (M) | weights (GiB) | KV cache per sequence (MiB) | 1 sequence (GiB) | 8 sequences (GiB) |
|---|---|---|---|---|---|---|
| 191M | Standard | 190.74 | 0.36 | 240 | 0.59 | 2.23 |
| | Loopback | 105.80 | 0.20 | 240 | 0.43 | 2.07 |
| 374M | Standard | 373.87 | 0.70 | 384 | 1.07 | 3.70 |
| | Loopback | 245.42 | 0.46 | 384 | 0.83 | 3.46 |
| 778M | Standard | 777.86 | 1.45 | 576 | 2.01 | 5.95 |
| | Loopback | 551.36 | 1.03 | 576 | 1.59 | 5.53 |
| 1.4B | Standard | 1,415.68 | 2.64 | 800 | 3.42 | 8.89 |
| | Loopback | 1,004.64 | 1.87 | 800 | 2.65 | 8.12 |
| 3B | Standard | 2,914.67 | 5.43 | 1,152 | 6.55 | 14.43 |
| | Loopback | 2,235.19 | 4.16 | 1,152 | 5.29 | 13.16 |

Loopback's stored block updates belong to the token being processed (K R vectors of width d, 24–72 KiB per token) and
are freed after its forward pass; they do not accumulate across tokens.
