| gate | verdict |
|---|---|
| B0 | PASS |
| B1 | PASS |
| B1b | PASS |
| B2 | PASS |
| B3 | PASS |
| B3c | FAIL |
| B4 | reported |

B0 interaction order (D=0, D=8): T1 1/1, T2 3/3, T3 4/4, T4 None/None
B1 lin on T1: 0.997
B1b first order on T2/T3: T2_lin 0.504, T2_pair 0.504, T3_lin 0.499, T3_pair 0.497
B2 no key (T4): lin 0.504, pair 0.504, mlp 0.501, branch 0.498, oracle 0.504
B3 branch: T2 0.994, exact subset 10/10, T3 0.992, exact subset 10/10
B3c cut: T2 repaired 0.565, T3 repaired 0.496, T3 key_removed 0.497

B4 training samples to reach 95% (grid up to 2048; - = not reached)

| task | D | branch | mlp |
|---|---|---|---|
| T2 | 0 | 32 | 32 |
| T2 | 8 | 64 | 256 |
| T2 | 16 | 64 | 1024 |
| T3 | 0 | 64 | 128 |
| T3 | 8 | 64 | 2048 |
| T3 | 16 | 128 | - |
