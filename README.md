# A3-PS003-TC442 · Team Nova
## NovaFlow: Federated Traffic-Flow Prediction with Communication-Cost Evaluation

**AIORI-3 Hackathon · Category: Smart Cities · Problem Statement PS-003**

| | |
|---|---|
| **Team** | Nova (A3-PS003-TC442) |
| **Members** | K. Nivedsagar · Stibin Steephen |
| **Mentor** | Dr. A. Immanuel Selvakumar |

### Project updates
- **Presentation:** uploaded to this repository as a PDF.
- **Live demo:** https://huggingface.co/spaces/nxvedsagar/novaflow-demo
- **Video presentation:** https://youtu.be/SrLrax792cI
- **Code and results:** simulator, 10 experiments × 3 seeds on the real dataset. Full report: [`reports/RESULTS.md`](reports/RESULTS.md)

---

## What this project does

Centralised traffic forecasting needs every roadside sensor to stream raw data to the cloud. **Federated Learning (FL)** instead trains a model at each junction and only sends model updates. This project builds a reproducible simulator that answers three questions on a real public dataset:

1. **Accuracy:** how close does Federated Averaging (FedAvg) get to a centralised model?
2. **Fairness:** what happens when junctions see very different traffic (non-IID), and who pays the price?
3. **Cost:** how many bytes does FL actually send, and when does it save bandwidth?

Our motto: **measure before you federate.**

## Key results (real data, mean of 3 seeds)

| Method | Test MAE | Test RMSE | Data sent |
|---|---|---|---|
| Persistence ("next = last" baseline) | 0.0266 | 0.0388 | – |
| **Centralised MLP** | **0.0220** | **0.0322** | – |
| FedAvg, IID junctions | 0.0219 | 0.0321 | 4.29 MB |
| FedAvg, non-IID junctions | 0.0258 | 0.0371 | 4.29 MB |
| Local-only (no collaboration) | 0.0225 | 0.0328 | 0 MB |
| PulseFed (our proposal) | 0.0269 | 0.0406 | 2.76 MB |
| PulseFed v2 (follow-up) | 0.0259 | 0.0378 | 2.79 MB |
| FedProx (μ = 0.01) | 0.0254 | 0.0366 | 4.29 MB |
| FedAvg + FP16 updates | 0.0257 | 0.0370 | 3.17 MB |
| FedAvg + INT8 updates | 0.0258 | 0.0371 | 2.74 MB |

Volumes are scaled 0–1 in the dataset, so an MAE of 0.022 means guesses are off by about 2.2% of the busiest traffic level.

**Main findings**
- With similar junctions, FL matches centralised accuracy while raw data stays local.
- With very different junctions, FedAvg is 17% worse than centralised, and the damage falls on the **quietest and busiest roads**. On the quietest roads it is even worse than the naive baseline.
- Training each road group alone beat the shared model here, which points to personalised FL as the next step.
- **INT8 compression cut traffic by 36% with no accuracy loss.**
- PulseFed, our heuristic that lets junctions skip rounds when their model barely changes, saved 37% of the bytes but hurt accuracy and starved the busiest roads.
- **PulseFed v2** (follow-up, rule fixed before running) reuses sleeping junctions' last update. It kept the saving and brought average accuracy back to FedAvg's level, but the busiest roads still suffer and results vary more across seeds; with 3 seeds the gain is promising, not proven. INT8 remains our recommendation.
- One FedAvg training sends about **24×** more data than the raw counts, equal to about **310 days** of streaming raw counts from all 36 sensors (our *Communication Payback Period*). For scalar sensors, FL's value is keeping data local; for cameras, the balance reverses.

See [`reports/RESULTS.md`](reports/RESULTS.md) for every number, per-seed values, per-junction results and all figures.

## Data-integrity audit

Before trusting the data, we audited it (`step2b_diagnose_layout.py`) and found three differences from the dataset's documentation:
1. The file has **48 columns, not 47**: 10 recent volumes (oldest to newest), a 5-category day indicator, 24 hour-of-day columns, number of lanes, 4 road-direction columns and a 4-column road name.
2. The provided target is **30 minutes** ahead, not 15. Because consecutive rows overlap, we rebuild the true 15-minute target (`TARGET_MODE` in `src/config.py`).
3. The documentation's dates suggest the test period comes first, but the data proves the training period flows directly into the test period, so the provided split **is** chronological (no data leakage).

## Dataset

[UCI Traffic Flow Forecasting](https://archive.ics.uci.edu/dataset/608/traffic+flow+forecasting) (id 608, CC BY 4.0): traffic volume every 15 minutes at 36 sensors on two highways in Northern Virginia / Washington D.C. Open Indian traffic datasets are still limited; the simulator is dataset-independent, so Indian ITMS data can be plugged in later.

Citation: Zhao, L. (2019). *Traffic Flow Forecasting* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C57897

---

## How to run

Requires Python 3.10–3.12. All commands run from the repository folder.

**Setup (once)**
```
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q                # expect: 34 passed
```

**Run the pipeline**
```
python step1_download_data.py      # download the dataset into data/raw/
python step2_inspect_data.py       # checks: layout, time order, target horizon
python step3_prepare_data.py       # features, chronological split, client preview
python step4_run_experiments.py    # 10 experiments x 3 seeds (about 1 minute)
python step5_make_plots.py         # figures/
python step6_make_report.py        # reports/RESULTS.md
```

**Quick smoke test (safe for demos, about 10 seconds)**
```
python step4_run_experiments.py --quick
python step5_make_plots.py --quick
python step6_make_report.py --quick
```
Quick runs go to `results_quick/`, `figures_quick/` and `reports/RESULTS_quick.md`, so they never touch the real results.

If Step 2 reports a problem, run `python step2b_diagnose_layout.py` for a detailed diagnosis.

## Experiments

| Name | What it is | Required by PS-003? |
|---|---|---|
| `centralized` | One model trained on all pooled training data | yes |
| `fedavg_iid` | FedAvg, rows shuffled evenly across 4 junctions | yes |
| `fedavg_noniid` | FedAvg, each junction owns a group of sensors (quiet → busy) | yes |
| `persistence` | Predict next volume = current volume | extra |
| `local_only_noniid` | Each junction trains alone | extra |
| `pulsefed_noniid` | PulseFed: only junctions whose model moved most talk (min 2, max 2 rounds asleep) | extra (our proposal) |
| `pulsefed_v2_noniid` | PulseFed v2: same selection, sleeping junctions' last update reused (no extra bytes) | extra (follow-up) |
| `fedprox_noniid` | FedAvg + proximal term | extra |
| `fedavg_noniid_fp16` / `_int8` | FedAvg with compressed uplink updates | extra |

**Fair comparison:** same model, features and test period for every method; centralised training gets the same number of passes over the data as FedAvg (30 rounds × 1 epoch); the test set is never used for tuning; seeds 0, 1, 2.

**Communication cost** is measured from the real serialised byte payloads (downlink, uplink, setup and control messages), not estimated.

## Project structure
```
├── step1_download_data.py      download dataset
├── step2_inspect_data.py       inspection + integrity checks
├── step2b_diagnose_layout.py   detailed layout / time-order diagnosis
├── step3_prepare_data.py       features, split, client preview
├── step4_run_experiments.py    all experiments x 3 seeds
├── step5_make_plots.py         figures
├── step6_make_report.py        reports/RESULTS.md
├── src/                        simulator: data, partitions, model, FedAvg + PulseFed, byte counting
├── tests/                      34 automated tests + synthetic dataset generator
├── docs/FEDAVG_PSEUDOCODE.md   FedAvg, PulseFed and PulseFed v2 pseudocode
├── results/  figures/  reports/   real-data outputs
└── .github/workflows/tests.yml automatic tests on every push
```

## Scope and limitations
Single-machine simulation, as required by PS-003: no live cameras, hardware, city-wide deployment or formal privacy guarantees. See section 9 of the report for the full list.

## References
- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017. https://proceedings.mlr.press/v54/mcmahan17a.html
- Li et al., *Federated Optimization in Heterogeneous Networks* (FedProx), MLSys 2020.
- Zhao, Gkountouna & Pfoser, *Spatial Auto-regressive Dependency Interpretable Learning Based on Spatial Topological Constraints*, ACM TSAS 2019. https://doi.org/10.1145/3339823
