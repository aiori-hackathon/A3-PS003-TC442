---
title: NovaFlow Federated Traffic Demo
emoji: 🚦
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Live federated traffic forecasting simulator
---

# NovaFlow · Team Nova · A3-PS003-TC442 (AIORI-3, PS-003)

Live demo of our federated traffic-flow prediction simulator. It runs the real Python code on the
UCI *Traffic Flow Forecasting* dataset and compares **centralised training, FedAvg, PulseFed and
PulseFed v2** on accuracy, fairness across junctions, and measured communication bytes.

- **Stage-by-stage tour:** an animated map of roads, junctions and the server shows every round live (model packets down, updates up, sleeping junctions), while all four approaches run one after another.
- **Run one method:** pick a method, number of rounds and seed.
- **Full results:** our pre-computed 3-seed, 30-round results and figures.

Team: K. Nivedsagar, Stibin Steephen · Mentor: Dr. A. Immanuel Selvakumar

Dataset: Zhao, L. (2019). *Traffic Flow Forecasting* [Dataset]. UCI Machine Learning Repository,
CC BY 4.0. https://doi.org/10.24432/C57897
