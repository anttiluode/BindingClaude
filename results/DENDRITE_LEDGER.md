| gate | verdict |
|---|---|
| D1 | PASS |
| D2 | PASS |
| D2b | PASS |
| D3 | PASS |
| D4b | PASS |
| D4 | PASS |

D1 exact: {'T2': {'max_single': 0.0, 'max_pair': 0.0, 'top': 1.0}, 'T3': {'max_single': 0.0, 'max_pair': 0.0, 'top': 1.0}}
D2 single-input Hebb on T1: 0.998
D2b blind: single_T2 0.498, single_T3 0.500, pair_T2 0.499, pair_T3 0.497
D3 dendrites: T2 mean 0.994, >=95% in 10/10, bound 10/10, T3 mean 0.992, >=95% in 10/10, bound 10/10
D4 linear dendrites: T2 0.493, T3 0.507
D4b knockout: T2 0.506, T3 0.495
H-and (reported): T2 0.982, bound 10/10, T3 0.495, bound 0/10

D5 experiences to 95% (median of seeds that reached it, stream 40,000)

| task | D | median | seeds reaching |
|---|---|---|---|
| T2 | 0 | 200.0 | 10/10 |
| T2 | 8 | 600.0 | 10/10 |
| T2 | 16 | 2200.0 | 10/10 |
| T3 | 0 | 200.0 | 10/10 |
| T3 | 8 | 1000.0 | 10/10 |
| T3 | 16 | 4600.0 | 10/10 |
