# The 72 registered predictions, by content and by timestamp

`scripts/prediction_split.py` (rules in its docstring; blind review of v0.9.11, item M2).

| content | held | failed | untested |
|---|---|---|---|
| construction | 5 | 0 | 0 |
| substantive | 48 | 18 | 1 |

| timestamp | held | failed | untested |
|---|---|---|---|
| dry_run | 9 | 0 | 1 |
| independent | 42 | 17 | 0 |
| local_only | 2 | 1 | 0 |

| content, timestamp | held | failed | untested |
|---|---|---|---|
| construction, independent | 5 | 0 | 0 |
| substantive, dry_run | 9 | 0 | 1 |
| substantive, independent | 37 | 17 | 0 |
| substantive, local_only | 2 | 1 | 0 |

With the exact halves of P5 and M9 counted as construction too, construction is 7 held of 7.

| table | id | content | timestamp | commit | outcome | note |
|---|---|---|---|---|---|---|
| D1 | P1 | substantive | independent | 9fb817d | failed |  |
| D1 | P2a | substantive | independent | 9fb817d | held |  |
| D1 | P2b | substantive | independent | 9fb817d | held |  |
| D1 | P3 | substantive | independent | 9fb817d | held |  |
| D1 | P4a | substantive | independent | 9fb817d | failed |  |
| D1 | P4b | substantive | independent | 9fb817d | held |  |
| D1 | P4c | substantive | independent | 9fb817d | held |  |
| D1 | P5 | substantive | independent | 9fb817d | held | Clopper-Pearson over in no cell (exact) and the strata gain nothing (substantive) |
| D1 | M1 | substantive | dry_run | d56feb6 | held |  |
| D1 | M2 | substantive | dry_run | d56feb6 | held |  |
| D1 | M3a | substantive | dry_run | d56feb6 | untested |  |
| D1 | M3b | substantive | dry_run | d56feb6 | held |  |
| D1 | M4 | substantive | dry_run | d56feb6 | held |  |
| D1 | M5 | substantive | dry_run | d56feb6 | held |  |
| D1 | M6 | substantive | dry_run | d56feb6 | held |  |
| D1 | M7 | substantive | dry_run | d56feb6 | held |  |
| D1 | M8 | substantive | dry_run | d56feb6 | held |  |
| D1 | M9 | substantive | dry_run | d56feb6 | held | Clopper-Pearson over in no cell (exact) and the strata gain nothing (substantive) |
| D2 | Q1 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | held |  |
| D2 | Q2 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | held |  |
| D2 | Q3 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | held |  |
| D2 | Q4 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | held |  |
| D2 | Q5 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | held |  |
| D2 | Q6 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | held |  |
| D2 | Y1 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | failed |  |
| D2 | Y2 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | held |  |
| D2 | Y3 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | held |  |
| D2 | Y4 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | held |  |
| D2 | Y5 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | failed |  |
| D2 | Y6 | construction | independent | bf7f5dc 1b1185f 3703288 56fbeda | held |  |
| D2 | Z1 | construction | independent | bf7f5dc 1b1185f 3703288 56fbeda | held |  |
| D2 | Z2 | construction | independent | bf7f5dc 1b1185f 3703288 56fbeda | held |  |
| D2 | Z3 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | held |  |
| D2 | Z4 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | failed |  |
| D2 | Z5 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | held |  |
| D2 | Z6 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | held |  |
| D2 | V1 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | failed |  |
| D2 | V2 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | failed |  |
| D2 | V3 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | held |  |
| D2 | V4 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | held |  |
| D2 | V5 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | held |  |
| D2 | V6 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | failed |  |
| D2 | S1 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | held |  |
| D2 | S2 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | held |  |
| D2 | S3 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | held |  |
| D2 | S4 | substantive | independent | bf7f5dc 1b1185f 3703288 56fbeda | held |  |
| D3 | T1 | substantive | local_only | cb50927 | held |  |
| D3 | T2 | substantive | local_only | cb50927 | held |  |
| D3 | T3 | substantive | local_only | cb50927 | failed |  |
| D4 | X1 | construction | independent | e3326b1 | held |  |
| D4 | X2 | substantive | independent | e3326b1 | held |  |
| D4 | X3 | substantive | independent | e3326b1 | held |  |
| D4 | X4 | substantive | independent | e3326b1 | failed |  |
| D4 | X5 | substantive | independent | e3326b1 | failed |  |
| D4 | X6 | substantive | independent | e3326b1 | failed |  |
| D4 | X7 | substantive | independent | e3326b1 | held |  |
| D4 | X8 | substantive | independent | e3326b1 | failed |  |
| D4 | W1 | construction | independent | e3326b1 | held |  |
| D4 | W2 | substantive | independent | e3326b1 | held |  |
| D4 | W3 | substantive | independent | e3326b1 | held |  |
| D4 | W4 | substantive | independent | e3326b1 | held |  |
| D4 | W5 | substantive | independent | e3326b1 | held |  |
| D4 | B4 Q1 | substantive | independent | e3326b1 | failed |  |
| D4 | B4 Q2 | substantive | independent | e3326b1 | failed |  |
| D4 | B4 Q3 | substantive | independent | e3326b1 | failed |  |
| D4 | B4 Q4 | substantive | independent | e3326b1 | held |  |
| D4 | B4 Q5 | substantive | independent | e3326b1 | failed |  |
| D4 | U1 | substantive | independent | e3326b1 | failed |  |
| D4 | U2 | substantive | independent | e3326b1 | held |  |
| D4 | U3 | substantive | independent | e3326b1 | held |  |
| D4 | U4 | substantive | independent | e3326b1 | held |  |
| D4 | U5 | substantive | independent | e3326b1 | held |  |
