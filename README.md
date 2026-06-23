# pneumapost — Station Operating Circular

*A pneumatic-dispatch network for sex-conditional lung-adenocarcinoma risk.*
Each clinical site is a **station**; the only thing that ever travels the tubes
is a sealed **carrier** holding sex-conditional class prototypes — never a raw
record, never a weight. The receiving exchange merges carriers, returns the
global prototype bank, and every station reads malignancy risk on-device.

This circular is the operating manual for that network: how to raise it, run a
dispatch cycle, and read the register against the reported figures.

---

## 1. What travels, and what stays

`pneumapost` implements **SexCond-FedProtoCon** (SC-PPC-FL): three modality
encoders feed a null-embedding gated fusion that places each patient on a unit
hypersphere; sex-conditional class prototypes `P(c,s)` are aligned contrastively
and synchronised by a prototype-only federated protocol.

| Tube | Package | Carries |
|------|---------|---------|
| intake | `src/pneumapost/intake` | tri-modal cohort, TCGA-LUAD / 16S adapters, missing-modality masking |
| apparatus | `src/pneumapost/apparatus` | omics graph-attention (~3.0M), 16S Set-Transformer (~1.9M), breath 1D-CNN, gated fusion Φ, sphere projection, prototype bank |
| franking | `src/pneumapost/franking` | prototypical-contrastive objective + sex-axis repulsion + null anchor + confidence shrinkage |
| dispatch | `src/pneumapost/dispatch` | federated training loop, confidence-weighted aggregation, empirical-Bayes shrinkage, FedProx proximal |
| delivery | `src/pneumapost/delivery` | on-device inference, temperature scaling, sensitivity-target calibration, reproduction harness |
| tallies | `src/pneumapost/tallies` | AUROC, Spec@95%Sens, balanced accuracy, F1, DeLong, bootstrap CI, Cohen's d, Holm–Bonferroni, BH–FDR |
| fittings | `src/pneumapost/fittings` | HOCON loader, seeding, hypersphere ops, logging |
| wicket | `src/pneumapost/wicket` | the service window: `train`, `appraise`, `predict`, `export` |

Run schedules live in `circulars/` (HOCON `.conf`, layered by `include`).

---

## 2. Raising the station

Pip:

```
python -m pip install -r requirements.txt
python -m pip install -e .
```

Conda:

```
conda env create -f environment.yml
conda activate pneumapost
python -m pip install -e .
```

Container (CUDA 11.3):

```
docker build -t pneumapost .
docker run --gpus all pneumapost appraise circulars/runs/main.conf
```

The stack is pinned to `torch==1.12.1+cu113`, `numpy==1.21.6`, Python 3.8. These
are deliberate; the CUDA 11.3 wheel index is required for the GPU build.

---

## 3. Goods received (data)

The private Layer-1 cohort is not redistributable, so the default cohort is a
synthetic tri-modal generator that reproduces the reported structure (two arms,
four sites, sex-conditional geometry, modality drop-out). Two public auxiliary
layers are wired for encoder pretraining and external comparison; both links are
verified reachable and recorded with access terms in **`datasets.txt`**:

- TCGA-LUAD multi-omics (GDC, open processed levels) — omics encoder.
- Oral / sputum 16S amplicon, ENA study SRP145042 — microbiome encoder.

Stage them with `dockets/prepare_data.sh`; the adapters read tab-separated
matrices and raise a clear error if a file is absent.

---

## 4. Tube schedule (training)

Primary result (paper compute profile):

```
python -m pneumapost.wicket train circulars/runs/main.conf --out runs/pneumapost.pt
```

Eight-GPU single-node dispatch:

```
bash dockets/launch_train.sh circulars/runs/main.conf runs/pneumapost.pt
```

Reported ablations and supplementary runs each have their own circular:

```
python -m pneumapost.wicket train circulars/runs/ablation_sex_agnostic.conf
python -m pneumapost.wicket train circulars/runs/ablation_no_repulsion.conf
python -m pneumapost.wicket train circulars/runs/ablation_no_null_anchor.conf
python -m pneumapost.wicket train circulars/runs/ablation_no_confidence.conf
python -m pneumapost.wicket train circulars/runs/ablation_breath_only.conf
python -m pneumapost.wicket train circulars/runs/ablation_microbiome_only.conf
python -m pneumapost.wicket train circulars/runs/ablation_omics_only.conf
python -m pneumapost.wicket train circulars/runs/supplementary_centralized_oracle.conf
python -m pneumapost.wicket train circulars/runs/supplementary_persite.conf
```

`circulars/runs/_bench.conf` is a shrunk schedule wired only into the test
bench; it is not a reporting configuration.

---

## 5. Dispatch register (evaluation)

```
python -m pneumapost.wicket appraise circulars/runs/main.conf --arm prospective
```

prints the two-arm comparison and the DeLong / bootstrap read-out against the
strongest federated competitor. On the default synthetic cohort the headline
figures land at:

| Read-out | Expected | Tolerance |
|----------|----------|-----------|
| SexCond-FedProtoCon, prospective female AUROC | 0.893 | ±0.012 |
| SexCond-FedProtoCon, never-smoker female AUROC | 0.901 | ±0.012 |
| SexCond-FedProtoCon, retrospective female AUROC | 0.908 | ±0.015 |
| Strongest federated baseline (DFL-MP) | ≈0.85 | gap ≥ 0.025 |
| Centralized tri-modal oracle (upper bound) | 0.901–0.915 | — |
| DeLong, proposed vs DFL-MP | p < 0.005 | — |

On-device reading (Algorithm 2) from a trained carrier:

```
python -m pneumapost.wicket predict runs/pneumapost.pt --limit 16
```

Edge export (ONNX reader for the e-nose SoC):

```
python -m pneumapost.wicket export runs/pneumapost.pt --out runs/pneumapost.onnx
```

---

## 6. Compute budget — read it before you commit

The reporting schedule (`circulars/runs/main.conf`) is sized for the paper's
training envelope and is heavy on purpose:

- world size **8** GPUs (80 GB class), `fp32`.
- per-GPU batch **512** × grad-accum **4** → effective batch **16384**.
- learning rate **5e-7**, 5000-step warmup, cosine decay.
- **20** federated rounds × **400** local epochs per site, **20** seeds.
- prototype carrier ≤ **1.8 MB** per round; no weights or raw records traverse.

Inference is the cheap half and is the deployment story: a tri-modal reading
fits a **4.7 MB** disk / **38 MB** RAM footprint and runs in **~41 ms** at INT8
on an ARM Cortex-A78-class edge SoC. Do not expect the training schedule to
finish on a workstation; the bench config exists for tests, not reporting.

---

## 7. Carriers (checkpoints)

`train` writes atomic checkpoints carrying the model state, the resolved run
spec and the seed, so a resumed run restores the same geometry. Released
carriers are available on reasonable request; verify any local carrier with
`sha256sum runs/pneumapost.pt` against the value distributed alongside it.
