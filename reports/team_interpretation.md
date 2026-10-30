1. **Federated learning matches centralised training when junctions are similar.** FedAvg on IID clients reached MAE 0.0219 versus 0.0220 for the centralised model, and came within 5% of centralised accuracy after only 7–8 rounds, without any raw data leaving the clients.

2. **Heterogeneity is the real cost, and it falls on the extremes.** With sensors grouped from quietest to busiest road (heterogeneity 0.437 vs 0.010 for IID), FedAvg was 17% worse than centralised and never reached the 5% target within 30 rounds. The per-client view shows where the damage lands: on the quietest roads (client 0) FedAvg was 31% worse than centralised and even worse than the naive persistence baseline (0.0226 vs 0.0196); on the busiest roads (client 3) it was 22% worse; client 1 was almost unaffected (about +2%). One averaged model fits the "average road" and neither extreme.

3. **Local models beat the shared model here.** Training each road group on its own data (MAE 0.0225) outperformed non-IID FedAvg (0.0258). Personalised federated learning, for example fine-tuning the global model locally, is the most promising next step.

4. **FedProx (μ = 0.01) gave no clear improvement** (0.0254 vs 0.0258; the gap is within seed-to-seed variation).

5. **Compression is a free win.** Sending INT8-quantised updates cut total traffic by 36% (4.29 → 2.74 MB) with no measurable accuracy loss; FP16 cut it by 26%.

6. **PulseFed, our proposed heuristic, did not pay off on this dataset.** It cut traffic by 37% but raised the error (MAE +4%, RMSE +9% versus FedAvg non-IID) and made results unstable across seeds. The equity view explains why: in the participation chart the busiest road group was repeatedly put to sleep and needed forced refreshes, and its error rose to 0.0365, worse than the persistence baseline (0.0332). INT8 achieves almost the same saving without this cost, so we recommend INT8 over PulseFed here. The overall MAE alone would have hidden this failure; the per-client check exposed it.

7. **Bandwidth: measure before you federate.** One FedAvg training run sends 4.29 MB, about 24 times the raw volume counts (0.18 MB) and equal to roughly 310 days of streaming raw counts from all 36 sensors. It is, however, less than uploading the full feature rows (5.63 MB). For scalar loop detectors, the value of federated learning lies in keeping data local rather than saving bandwidth; for data-heavy sensors such as cameras, the balance reverses.
