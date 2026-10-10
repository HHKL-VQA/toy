# Results

### 2x3, seed 0, t = 0.05

| L | CNOT | depth | params | eps_spec (Trotter) | eps_spec (trained) | improvement | eps_spec / eps_blk | eps_rHS (trained) | spectral / rHS |
|---|---|---|---|---|---|---|---|---|---|
| 2 (S2^1) | 42 | 18 | 17 | 4.122e-03 | 3.249e-03 | 1.27 | — | 1.087e-03 | 2.99 |
| 10 (S4) | 174 | 78 | 77 | 7.574e-06 | 3.437e-06 | 2.20 | — | 9.892e-07 | 3.47 |

| L | J0 | J final | iters | stop | drift | held-out diff | cert its (init / trained) | starts agree | wall |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 9.491e-07 | 5.906e-07 | 33 | plateau | 1.4e-15 | 3.25 % | 21,21 / 28,28 | True/True | 0.00 h |
| 10 | 2.517e-12 | 4.893e-13 | 211 | plateau | 4.5e-15 | 25.87 % | 20,20 / 23,23 | True/True | 0.01 h |
