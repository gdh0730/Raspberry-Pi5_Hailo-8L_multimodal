# Phase36 FP32 vs Hailo (Best5)

- done: 4
- pending: 1

| Track | Mode | Status | FP32 F1 | Hailo F1 | Delta F1(H-F) | FP32 Acc | Hailo Acc | Delta Acc(H-F) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ood_c2r | fusion | done | 0.1080 | 0.0550 | -0.0530 | 0.2216 | 0.1837 | -0.0379 |
| id_crema | fusion | done | 0.6643 | 0.6162 | -0.0482 | 0.6653 | 0.6247 | -0.0407 |
| id_ravdess | fusion | done | 0.5811 | 0.5346 | -0.0465 | 0.5864 | 0.5409 | -0.0455 |
| id_all | fusion | done | 0.6261 | 0.5892 | -0.0369 | 0.6290 | 0.5935 | -0.0355 |
| ood_r2c | fusion | pending | 0.1278 | - | - | 0.2231 | - | - |
