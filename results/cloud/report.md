# Results

### 2x3, seed 0, t = 0.05

| L | CNOT | depth | params | eps_spec (Trotter) | eps_spec (trained) | improvement | eps_spec / eps_blk | eps_rHS (trained) | spectral / rHS |
|---|---|---|---|---|---|---|---|---|---|
| 2 (S2^1) | 42 | 18 | 17 | 4.122e-03 | 3.249e-03 | 1.27 | - | 1.087e-03 | 2.99 |
| 10 (S4) | 174 | 78 | 77 | 7.574e-06 | 1.946e-06 | 3.89 | - | 5.994e-07 | 3.25 |

| L | J0 | J final | iters | stop | drift | held-out diff | cert its (init / trained) | starts agree | wall |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 9.491e-07 | 5.906e-07 | 7 | scipy | 1.3e-15 | 3.24 % | 21,21 / 28,25 | True/True | 0.00 h |
| 10 | 2.517e-12 | 1.796e-13 | 500 | maxiter 500 | 4.5e-15 | 35.82 % | 19,20 / 28,27 | True/True | 0.01 h |
