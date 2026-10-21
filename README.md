# CircMTL

Multi-task graph learning for circRNA association prediction, evaluated under a strict, pre-registered protocol.

CircMTL predicts circRNA–disease associations on a heterogeneous graph and asks one question: does adding circRNA–miRNA and circRNA–RBP interactions as auxiliary tasks help the disease task? Three models answer it on identical folds, negatives and training settings:

| Model | Graph | Loss |
|---|---|---|
| M1 | circRNA–disease only | disease |
| M2 | all relations | disease only (auxiliary edges as input) |
| M3 | all relations | disease + RBP + miRNA |

The encoder is a 2-layer R-GCN with a DistMult decoder per relation. circRNA inputs are RNA-FM layer-12 embeddings (640 dimensions) plus log10 length. HGT is tested as a sensitivity analysis only.

This is a master's project at Queen's University (MIB Lab). The repository holds the code, the frozen configs and the notebooks that produced the results. It does not hold the data or the pretrained weights (see [Data](#data)).

## What the project found

The main result is deliberately modest, and the repository is built so that it can be checked.

- Under host-gene folds with degree-matched negatives, disease prediction stays close to the trivial baselines: log10 length alone gives an AUROC of 0.527, and a disease-popularity baseline gives 0.510.
- None of the disease-side additions moved that. MeSH similarity features changed M3 by +0.003 (95% CI −0.007 to 0.012); a host-gene prior from CTD scored 0.507; an HMDD miRNA enrichment score scored 0.509; a circAtlas tissue profile scored 0.542 against a pass mark of 0.55. No tested disease-side input exceeded 0.534 on degree-matched negatives apart from the tissue diagnostic, which also missed its rule.
- HGT did not replace R-GCN (M3_hgt − M3 = −0.002), and removing the length feature changed M3 by +0.002.
- A **protocol ladder** (runs A, B, C, C2) relaxes one protocol choice at a time (pair-level splits, uniform negatives, label-containing features) to show how much of a published-style score comes from the evaluation design rather than the model.
- **Run X** scores the models on an external circRNADisease v2 file (642 positives). The models land near their strict-protocol level and below the popularity floor, and the multi-task difference cannot be resolved.

<!-- TODO: add the M1/M2/M3 degree-matched disease AUROC table (mean over 5 folds x 3 seeds, with bootstrap CIs) from notebooks/Final_notebook/08_train.ipynb. -->

## Pre-registration

Every analysis has a YAML file in `configs/` that was written before its results existed. Later amendments are labelled as such inside the file, with what was seen before each amendment. Pass rules, floors and stop rules are in those files, not in the notebooks.

The git history is part of the record. Two tags mark the main points:

- `v1-strict-results`: the frozen strict-protocol results (S0) and run B, with file hashes in `results_manifest.txt`
- `ladder-registered`: the same commit, from which the ladder runs were registered and executed

To read the hash record of the strict runs:

```powershell
git show v1-strict-results:results_manifest.txt
```

The commit messages note which runs were executed once, which were repeated (for example, the second execution of run B) and which analyses were exploratory.

## Repository layout

```
configs/      frozen specifications (graph, model, probe, extensions, protocol ladder)
src/circmtl/  graph construction, similarity edges, ladder helpers
notebooks/Final_notebook/
              05_probe              Objective 1: linear probe on RNA-FM embeddings
              06_embeddings         RNA-FM embeddings (overlapping and non-overlapping windows, layer 6 and 12)
              07_graph              graph tables, host-gene folds, frozen negatives
              08_train              M1, M2, M3 and the sensitivity analyses (HGT, no length, MeSH)
              09_disease_extension  MeSH features and the training-free diagnostics
              10_protocol_ladder    runs A, B, C, C2, S0, run X and the enhancement round
figure_cells/ report and figure code for the ladder, run X and the ranking analysis
scripts/      small checks (registration, floor verification)
```

`configs/` map to the analyses as follows:

| File | Covers |
|---|---|
| `probe.yaml` | Objective 1 linear-probe protocol |
| `graph.yaml` | graph schema, splits, models, baselines, model selection |
| `hgt.yaml`, `sensitivity_nolen.yaml` | HGT and no-length sensitivities |
| `graph_ext.yaml` | Extension A: MeSH features, host-gene prior, HMDD enrichment |
| `expression.yaml` | Extension B: circAtlas tissue profiles |
| `extension.yaml` | Extension E1: Open Targets host-gene prior |
| `protocol_ladder.yaml` | protocol ladder, enhancement round 1, run X |
| `model.yaml` | superseded per-target classification design (kept as a record) |

## Setup

The project was developed on Windows with PowerShell, Python 3.11 and an NVIDIA GPU (CUDA 12.6 wheels).

```powershell
conda env create -f environment.yml
conda activate circmtl
pip install -e .
```

`environment.yml` installs PyTorch from the CUDA 12.6 index. On a machine without that GPU, change the `--extra-index-url` line to the matching PyTorch build.

Paths are set in `configs/paths.yaml` relative to the project root.

## Data

Raw and processed data are not included. The notebooks expect this layout under `data/`:

| Folder | Source |
|---|---|
| `raw/CircR2Disease_V2.0` | CircR2Disease v2.0 (circRNA–disease entries, circRNA list, sequences) |
| `raw/encori` | ENCORI/starBase hg38 miRNA–circRNA and RBP–circRNA tables |
| `raw/ncbi_grch38` | NCBI RefSeq GRCh38 genome, GCF_000001405.26 |
| `raw/mesh` | MeSH 2026 descriptors (`desc2026.xml`) |
| `raw/hmdd` | HMDD v4.0 (`alldata_v4.txt`) |
| `raw/ctd` | CTD gene–disease associations |
| `raw/circatlas` | circAtlas 3.0 human tissue exports and ID conversion |
| `raw/circRNADisese_V2` | circRNADisease v2 (external set for run X) |

The ENCORI reference is Zhou et al., Nature Methods 2026 (doi 10.1038/s41592-026-03105-x).

Processed tables are built by the notebooks in order, starting from the exploration and filtering steps and ending at `07_graph`. Frozen folds and negatives are written once and reused by every model.

## Pretrained models

Neither model is included.

- **RNA-FM**: download the pretrained weights from the official RNA-FM release and place them at `models/RNA-FM_pretrained.pth`.
- **circRNA2Vec**: an alternative circRNA embedding model with its own environment (`environment_circ2vec.yml`); place its files under `models/circRNA2Vec/`.

## Citation and license

<!-- TODO: add the thesis or paper citation, a license (for example MIT) and a contact line. -->
