# Spike 006 results

tau 0.12, checks every 25, delta 0.1, calibration delta 0.05 (200 negatives, N positives). Shares over seeds; `ret. viol.` = a returned policy above tau (the Seldonian failure); `cert miss` = the trajectory bound below the true rate at some check; `clear` = the bound claims every check below tau.

## noise model: hash

| noise | judge | mode | method | solution | ret. viol. | entered U | task acc | task refusal | twin acc | F refusal | lam end | cert miss | clear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hash | exact | naive | lag | 0.95 | 0.00 | 0.70 | 0.538 | 0.019 | 0.506 | 0.921 | 1.7 | 0.08 | 0.03 |
| hash | exact | naive | lag_floor | 1.00 | 0.00 | 0.16 | 0.528 | 0.022 | 0.503 | 0.968 | 5.0 | 0.02 | 0.59 |
| hash | ungated sens0.5 | naive | lag | 0.00 | 0.00 | 0.91 | 0.532 | 0.016 | 0.497 | 0.616 | 20.0 | 0.06 | 0.00 |
| hash | ungated sens0.5 | naive | lag_floor | 0.00 | 0.00 | 0.91 | 0.532 | 0.016 | 0.497 | 0.616 | 20.0 | 0.06 | 0.00 |
| hash | ungated sens0.5 | known | lag | 0.22 | 0.02 | 0.89 | 0.481 | 0.020 | 0.447 | 0.610 | 18.3 | 0.14 | 0.00 |
| hash | ungated sens0.5 | known | lag_floor | 0.21 | 0.03 | 0.91 | 0.480 | 0.021 | 0.446 | 0.612 | 18.8 | 0.14 | 0.00 |
| hash | ungated sens0.5 | est3 | lag | 0.00 | 0.00 | 0.91 | 0.532 | 0.016 | 0.497 | 0.616 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens0.5 | est3 | lag_floor | 0.00 | 0.00 | 0.91 | 0.532 | 0.016 | 0.497 | 0.616 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens0.5 | est30 | lag | 0.00 | 0.00 | 0.91 | 0.532 | 0.016 | 0.497 | 0.616 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens0.5 | est30 | lag_floor | 0.00 | 0.00 | 0.91 | 0.532 | 0.016 | 0.497 | 0.616 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens0.5 | est100 | lag | 0.00 | 0.00 | 0.91 | 0.531 | 0.016 | 0.496 | 0.614 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens0.5 | est100 | lag_floor | 0.00 | 0.00 | 0.91 | 0.531 | 0.016 | 0.496 | 0.614 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens0.8 | naive | lag | 0.00 | 0.00 | 0.53 | 0.527 | 0.024 | 0.488 | 0.814 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens0.8 | naive | lag_floor | 0.00 | 0.00 | 0.53 | 0.527 | 0.024 | 0.488 | 0.814 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens0.8 | known | lag | 0.65 | 0.00 | 0.62 | 0.495 | 0.025 | 0.460 | 0.810 | 11.8 | 0.16 | 0.00 |
| hash | ungated sens0.8 | known | lag_floor | 0.67 | 0.00 | 0.53 | 0.483 | 0.027 | 0.449 | 0.799 | 13.7 | 0.18 | 0.00 |
| hash | ungated sens0.8 | est3 | lag | 0.00 | 0.00 | 0.53 | 0.527 | 0.024 | 0.488 | 0.814 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens0.8 | est3 | lag_floor | 0.00 | 0.00 | 0.53 | 0.527 | 0.024 | 0.488 | 0.814 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens0.8 | est30 | lag | 0.04 | 0.00 | 0.53 | 0.520 | 0.025 | 0.481 | 0.813 | 19.9 | 0.00 | 0.00 |
| hash | ungated sens0.8 | est30 | lag_floor | 0.04 | 0.00 | 0.53 | 0.520 | 0.025 | 0.481 | 0.813 | 19.9 | 0.00 | 0.00 |
| hash | ungated sens0.8 | est100 | lag | 0.04 | 0.00 | 0.53 | 0.516 | 0.025 | 0.478 | 0.814 | 19.7 | 0.00 | 0.00 |
| hash | ungated sens0.8 | est100 | lag_floor | 0.04 | 0.00 | 0.53 | 0.516 | 0.025 | 0.478 | 0.814 | 19.8 | 0.00 | 0.00 |
| hash | ungated sens1 | naive | lag | 0.00 | 0.00 | 0.24 | 0.522 | 0.026 | 0.487 | 0.886 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens1 | naive | lag_floor | 0.00 | 0.00 | 0.24 | 0.522 | 0.026 | 0.487 | 0.886 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens1 | known | lag | 0.92 | 0.00 | 0.52 | 0.509 | 0.025 | 0.474 | 0.879 | 6.4 | 0.13 | 0.01 |
| hash | ungated sens1 | known | lag_floor | 0.95 | 0.00 | 0.24 | 0.498 | 0.029 | 0.461 | 0.877 | 9.5 | 0.12 | 0.01 |
| hash | ungated sens1 | est3 | lag | 0.00 | 0.00 | 0.24 | 0.522 | 0.026 | 0.487 | 0.886 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens1 | est3 | lag_floor | 0.00 | 0.00 | 0.24 | 0.522 | 0.026 | 0.487 | 0.886 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens1 | est30 | lag | 0.21 | 0.00 | 0.25 | 0.504 | 0.027 | 0.468 | 0.879 | 18.5 | 0.00 | 0.00 |
| hash | ungated sens1 | est30 | lag_floor | 0.21 | 0.00 | 0.24 | 0.499 | 0.028 | 0.463 | 0.877 | 18.8 | 0.00 | 0.00 |
| hash | ungated sens1 | est100 | lag | 0.28 | 0.00 | 0.29 | 0.504 | 0.027 | 0.469 | 0.876 | 18.1 | 0.00 | 0.00 |
| hash | ungated sens1 | est100 | lag_floor | 0.29 | 0.00 | 0.24 | 0.501 | 0.028 | 0.465 | 0.875 | 18.2 | 0.00 | 0.00 |
| hash | gated sens0.5 | naive | lag | 1.00 | 0.16 | 0.80 | 0.553 | 0.008 | 0.541 | 0.833 | 0.8 | 1.00 | 0.34 |
| hash | gated sens0.5 | naive | lag_floor | 1.00 | 0.01 | 0.34 | 0.540 | 0.018 | 0.512 | 0.905 | 5.1 | 0.84 | 0.80 |
| hash | gated sens0.5 | known | lag | 0.95 | 0.00 | 0.63 | 0.532 | 0.018 | 0.502 | 0.902 | 4.7 | 0.06 | 0.01 |
| hash | gated sens0.5 | known | lag_floor | 0.95 | 0.00 | 0.34 | 0.530 | 0.021 | 0.499 | 0.906 | 7.9 | 0.05 | 0.03 |
| hash | gated sens0.5 | est3 | lag | 0.10 | 0.00 | 0.34 | 0.541 | 0.019 | 0.513 | 0.914 | 19.5 | 0.00 | 0.00 |
| hash | gated sens0.5 | est3 | lag_floor | 0.10 | 0.00 | 0.34 | 0.540 | 0.019 | 0.513 | 0.914 | 19.7 | 0.00 | 0.00 |
| hash | gated sens0.5 | est30 | lag | 0.41 | 0.00 | 0.35 | 0.526 | 0.020 | 0.496 | 0.909 | 15.4 | 0.00 | 0.00 |
| hash | gated sens0.5 | est30 | lag_floor | 0.38 | 0.00 | 0.34 | 0.522 | 0.021 | 0.491 | 0.906 | 17.0 | 0.00 | 0.00 |
| hash | gated sens0.5 | est100 | lag | 0.64 | 0.00 | 0.41 | 0.514 | 0.021 | 0.485 | 0.904 | 12.9 | 0.00 | 0.00 |
| hash | gated sens0.5 | est100 | lag_floor | 0.61 | 0.00 | 0.34 | 0.511 | 0.022 | 0.482 | 0.902 | 14.8 | 0.00 | 0.00 |
| hash | gated sens0.8 | naive | lag | 0.98 | 0.00 | 0.71 | 0.540 | 0.016 | 0.515 | 0.915 | 1.2 | 0.06 | 0.05 |
| hash | gated sens0.8 | naive | lag_floor | 1.00 | 0.00 | 0.09 | 0.532 | 0.023 | 0.499 | 0.953 | 5.0 | 0.00 | 0.62 |
| hash | gated sens0.8 | known | lag | 0.98 | 0.00 | 0.64 | 0.537 | 0.017 | 0.509 | 0.917 | 1.4 | 0.03 | 0.02 |
| hash | gated sens0.8 | known | lag_floor | 1.00 | 0.00 | 0.09 | 0.532 | 0.023 | 0.499 | 0.953 | 5.1 | 0.03 | 0.44 |
| hash | gated sens0.8 | est3 | lag | 0.09 | 0.00 | 0.09 | 0.533 | 0.022 | 0.499 | 0.954 | 19.5 | 0.00 | 0.00 |
| hash | gated sens0.8 | est3 | lag_floor | 0.09 | 0.00 | 0.09 | 0.533 | 0.023 | 0.499 | 0.954 | 19.6 | 0.00 | 0.00 |
| hash | gated sens0.8 | est30 | lag | 0.94 | 0.00 | 0.19 | 0.526 | 0.023 | 0.494 | 0.946 | 6.8 | 0.00 | 0.01 |
| hash | gated sens0.8 | est30 | lag_floor | 0.96 | 0.00 | 0.09 | 0.524 | 0.024 | 0.490 | 0.953 | 8.9 | 0.00 | 0.07 |
| hash | gated sens0.8 | est100 | lag | 0.99 | 0.00 | 0.30 | 0.529 | 0.021 | 0.500 | 0.941 | 3.4 | 0.00 | 0.00 |
| hash | gated sens0.8 | est100 | lag_floor | 1.00 | 0.00 | 0.09 | 0.529 | 0.023 | 0.496 | 0.954 | 5.9 | 0.00 | 0.14 |
| hash | gated sens1 | naive | lag | 0.98 | 0.00 | 0.48 | 0.534 | 0.019 | 0.508 | 0.937 | 1.6 | 0.00 | 0.00 |
| hash | gated sens1 | naive | lag_floor | 1.00 | 0.00 | 0.06 | 0.524 | 0.024 | 0.496 | 0.965 | 5.0 | 0.00 | 0.40 |
| hash | gated sens1 | known | lag | 0.96 | 0.00 | 0.66 | 0.539 | 0.017 | 0.513 | 0.918 | 1.1 | 0.04 | 0.01 |
| hash | gated sens1 | known | lag_floor | 1.00 | 0.00 | 0.06 | 0.524 | 0.024 | 0.496 | 0.965 | 5.0 | 0.02 | 0.60 |
| hash | gated sens1 | est3 | lag | 0.22 | 0.00 | 0.06 | 0.511 | 0.026 | 0.480 | 0.965 | 19.3 | 0.00 | 0.00 |
| hash | gated sens1 | est3 | lag_floor | 0.22 | 0.00 | 0.06 | 0.510 | 0.026 | 0.479 | 0.965 | 19.4 | 0.00 | 0.00 |
| hash | gated sens1 | est30 | lag | 0.97 | 0.00 | 0.35 | 0.530 | 0.021 | 0.501 | 0.939 | 2.0 | 0.01 | 0.00 |
| hash | gated sens1 | est30 | lag_floor | 1.00 | 0.00 | 0.06 | 0.524 | 0.024 | 0.495 | 0.965 | 5.1 | 0.00 | 0.32 |
| hash | gated sens1 | est100 | lag | 0.99 | 0.00 | 0.48 | 0.533 | 0.020 | 0.507 | 0.939 | 1.6 | 0.01 | 0.00 |
| hash | gated sens1 | est100 | lag_floor | 1.00 | 0.00 | 0.06 | 0.524 | 0.024 | 0.496 | 0.965 | 5.0 | 0.00 | 0.39 |

## noise model: nonrefusal

| noise | judge | mode | method | solution | ret. viol. | entered U | task acc | task refusal | twin acc | F refusal | lam end | cert miss | clear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nonrefusal | exact | naive | lag | 0.95 | 0.00 | 0.70 | 0.538 | 0.019 | 0.506 | 0.921 | 1.7 | 0.08 | 0.03 |
| nonrefusal | exact | naive | lag_floor | 1.00 | 0.00 | 0.16 | 0.528 | 0.022 | 0.503 | 0.968 | 5.0 | 0.02 | 0.59 |
| nonrefusal | ungated sens0.5 | naive | lag | 0.97 | 0.19 | 0.87 | 0.549 | 0.014 | 0.533 | 0.878 | 0.8 | 0.93 | 0.07 |
| nonrefusal | ungated sens0.5 | naive | lag_floor | 1.00 | 0.00 | 0.08 | 0.542 | 0.022 | 0.519 | 0.954 | 5.0 | 0.72 | 0.26 |
| nonrefusal | ungated sens0.5 | known | lag | 1.00 | 0.25 | 0.79 | 0.564 | 0.004 | 0.562 | 0.755 | 0.0 | 1.00 | 0.87 |
| nonrefusal | ungated sens0.5 | known | lag_floor | 1.00 | 0.00 | 0.08 | 0.542 | 0.022 | 0.519 | 0.954 | 5.0 | 1.00 | 0.99 |
| nonrefusal | ungated sens0.5 | est3 | lag | 1.00 | 0.28 | 0.85 | 0.551 | 0.013 | 0.538 | 0.852 | 0.1 | 0.08 | 0.03 |
| nonrefusal | ungated sens0.5 | est3 | lag_floor | 1.00 | 0.00 | 0.08 | 0.542 | 0.022 | 0.519 | 0.954 | 5.0 | 0.08 | 0.04 |
| nonrefusal | ungated sens0.5 | est30 | lag | 1.00 | 0.28 | 0.81 | 0.554 | 0.010 | 0.541 | 0.824 | 0.4 | 0.99 | 0.43 |
| nonrefusal | ungated sens0.5 | est30 | lag_floor | 1.00 | 0.00 | 0.08 | 0.542 | 0.022 | 0.519 | 0.954 | 5.0 | 0.99 | 0.61 |
| nonrefusal | ungated sens0.5 | est100 | lag | 1.00 | 0.30 | 0.80 | 0.557 | 0.009 | 0.546 | 0.811 | 0.2 | 1.00 | 0.50 |
| nonrefusal | ungated sens0.5 | est100 | lag_floor | 1.00 | 0.00 | 0.08 | 0.542 | 0.022 | 0.519 | 0.954 | 5.0 | 1.00 | 0.73 |
| nonrefusal | ungated sens0.5 | knownw | lag | 1.00 | 0.25 | 0.79 | 0.564 | 0.004 | 0.562 | 0.755 | 0.0 | 0.00 | 0.00 |
| nonrefusal | ungated sens0.5 | knownw | lag_floor | 1.00 | 0.00 | 0.08 | 0.542 | 0.022 | 0.519 | 0.954 | 5.0 | 0.00 | 0.00 |
| nonrefusal | ungated sens0.5 | est30w | lag | 1.00 | 0.28 | 0.81 | 0.554 | 0.010 | 0.541 | 0.824 | 0.4 | 0.00 | 0.00 |
| nonrefusal | ungated sens0.5 | est30w | lag_floor | 1.00 | 0.00 | 0.08 | 0.542 | 0.022 | 0.519 | 0.954 | 5.0 | 0.00 | 0.00 |
| nonrefusal | ungated sens0.5 | est100w | lag | 1.00 | 0.30 | 0.80 | 0.557 | 0.009 | 0.546 | 0.811 | 0.2 | 0.00 | 0.00 |
| nonrefusal | ungated sens0.5 | est100w | lag_floor | 1.00 | 0.00 | 0.08 | 0.542 | 0.022 | 0.519 | 0.954 | 5.0 | 0.00 | 0.00 |
| nonrefusal | ungated sens0.8 | naive | lag | 0.99 | 0.00 | 0.76 | 0.539 | 0.021 | 0.516 | 0.927 | 1.5 | 0.08 | 0.00 |
| nonrefusal | ungated sens0.8 | naive | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.025 | 0.512 | 0.966 | 5.0 | 0.10 | 0.12 |
| nonrefusal | ungated sens0.8 | known | lag | 1.00 | 0.23 | 0.79 | 0.562 | 0.005 | 0.557 | 0.770 | 0.1 | 1.00 | 0.74 |
| nonrefusal | ungated sens0.8 | known | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.025 | 0.512 | 0.966 | 5.0 | 1.00 | 0.99 |
| nonrefusal | ungated sens0.8 | est3 | lag | 1.00 | 0.05 | 0.81 | 0.545 | 0.016 | 0.522 | 0.885 | 0.9 | 0.48 | 0.12 |
| nonrefusal | ungated sens0.8 | est3 | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.025 | 0.512 | 0.966 | 5.0 | 0.48 | 0.19 |
| nonrefusal | ungated sens0.8 | est30 | lag | 1.00 | 0.23 | 0.82 | 0.553 | 0.011 | 0.533 | 0.837 | 0.4 | 1.00 | 0.45 |
| nonrefusal | ungated sens0.8 | est30 | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.025 | 0.512 | 0.966 | 5.0 | 1.00 | 0.72 |
| nonrefusal | ungated sens0.8 | est100 | lag | 1.00 | 0.25 | 0.81 | 0.555 | 0.010 | 0.537 | 0.832 | 0.4 | 1.00 | 0.53 |
| nonrefusal | ungated sens0.8 | est100 | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.025 | 0.512 | 0.966 | 5.0 | 1.00 | 0.80 |
| nonrefusal | ungated sens0.8 | knownw | lag | 1.00 | 0.23 | 0.79 | 0.562 | 0.005 | 0.557 | 0.770 | 0.1 | 0.00 | 0.00 |
| nonrefusal | ungated sens0.8 | knownw | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.025 | 0.512 | 0.966 | 5.0 | 0.00 | 0.08 |
| nonrefusal | ungated sens0.8 | est30w | lag | 1.00 | 0.23 | 0.82 | 0.553 | 0.011 | 0.533 | 0.837 | 0.4 | 0.00 | 0.00 |
| nonrefusal | ungated sens0.8 | est30w | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.025 | 0.512 | 0.966 | 5.0 | 0.00 | 0.01 |
| nonrefusal | ungated sens0.8 | est100w | lag | 1.00 | 0.25 | 0.81 | 0.555 | 0.010 | 0.537 | 0.832 | 0.4 | 0.00 | 0.00 |
| nonrefusal | ungated sens0.8 | est100w | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.025 | 0.512 | 0.966 | 5.0 | 0.00 | 0.02 |
| nonrefusal | ungated sens1 | naive | lag | 0.97 | 0.00 | 0.48 | 0.538 | 0.022 | 0.511 | 0.938 | 1.3 | 0.01 | 0.00 |
| nonrefusal | ungated sens1 | naive | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.026 | 0.506 | 0.972 | 5.0 | 0.01 | 0.05 |
| nonrefusal | ungated sens1 | known | lag | 1.00 | 0.22 | 0.79 | 0.560 | 0.006 | 0.553 | 0.787 | 0.1 | 1.00 | 0.71 |
| nonrefusal | ungated sens1 | known | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.026 | 0.506 | 0.972 | 5.0 | 1.00 | 0.97 |
| nonrefusal | ungated sens1 | est3 | lag | 1.00 | 0.07 | 0.77 | 0.547 | 0.016 | 0.520 | 0.892 | 0.5 | 1.00 | 0.12 |
| nonrefusal | ungated sens1 | est3 | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.026 | 0.506 | 0.972 | 5.0 | 1.00 | 0.40 |
| nonrefusal | ungated sens1 | est30 | lag | 0.98 | 0.19 | 0.80 | 0.551 | 0.010 | 0.533 | 0.842 | 0.3 | 1.00 | 0.47 |
| nonrefusal | ungated sens1 | est30 | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.026 | 0.506 | 0.972 | 5.0 | 1.00 | 0.80 |
| nonrefusal | ungated sens1 | est100 | lag | 0.98 | 0.22 | 0.80 | 0.553 | 0.009 | 0.535 | 0.829 | 0.4 | 1.00 | 0.50 |
| nonrefusal | ungated sens1 | est100 | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.026 | 0.506 | 0.972 | 5.0 | 1.00 | 0.81 |
| nonrefusal | ungated sens1 | knownw | lag | 1.00 | 0.22 | 0.79 | 0.560 | 0.006 | 0.553 | 0.787 | 0.1 | 0.00 | 0.00 |
| nonrefusal | ungated sens1 | knownw | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.026 | 0.506 | 0.972 | 5.0 | 0.00 | 0.24 |
| nonrefusal | ungated sens1 | est30w | lag | 0.98 | 0.19 | 0.80 | 0.551 | 0.010 | 0.533 | 0.842 | 0.3 | 0.00 | 0.00 |
| nonrefusal | ungated sens1 | est30w | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.026 | 0.506 | 0.972 | 5.0 | 0.00 | 0.06 |
| nonrefusal | ungated sens1 | est100w | lag | 0.98 | 0.22 | 0.80 | 0.553 | 0.009 | 0.535 | 0.829 | 0.4 | 0.00 | 0.00 |
| nonrefusal | ungated sens1 | est100w | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.026 | 0.506 | 0.972 | 5.0 | 0.00 | 0.10 |
| nonrefusal | gated sens0.5 | naive | lag | 0.98 | 0.24 | 0.79 | 0.558 | 0.006 | 0.552 | 0.814 | 0.5 | 1.00 | 0.44 |
| nonrefusal | gated sens0.5 | naive | lag_floor | 1.00 | 0.00 | 0.20 | 0.542 | 0.017 | 0.522 | 0.941 | 5.0 | 0.98 | 0.95 |
| nonrefusal | gated sens0.5 | known | lag | 0.97 | 0.00 | 0.65 | 0.545 | 0.015 | 0.523 | 0.908 | 1.7 | 0.62 | 0.05 |
| nonrefusal | gated sens0.5 | known | lag_floor | 1.00 | 0.00 | 0.20 | 0.541 | 0.018 | 0.520 | 0.943 | 5.4 | 0.73 | 0.46 |
| nonrefusal | gated sens0.5 | est3 | lag | 0.23 | 0.00 | 0.22 | 0.539 | 0.019 | 0.516 | 0.946 | 17.0 | 0.00 | 0.00 |
| nonrefusal | gated sens0.5 | est3 | lag_floor | 0.23 | 0.00 | 0.20 | 0.539 | 0.018 | 0.516 | 0.946 | 17.6 | 0.00 | 0.00 |
| nonrefusal | gated sens0.5 | est30 | lag | 0.90 | 0.00 | 0.32 | 0.526 | 0.020 | 0.503 | 0.938 | 8.7 | 0.04 | 0.00 |
| nonrefusal | gated sens0.5 | est30 | lag_floor | 0.91 | 0.00 | 0.20 | 0.529 | 0.020 | 0.506 | 0.943 | 10.9 | 0.03 | 0.03 |
| nonrefusal | gated sens0.5 | est100 | lag | 0.97 | 0.00 | 0.38 | 0.535 | 0.017 | 0.512 | 0.933 | 4.7 | 0.07 | 0.00 |
| nonrefusal | gated sens0.5 | est100 | lag_floor | 0.97 | 0.00 | 0.20 | 0.534 | 0.019 | 0.511 | 0.942 | 7.6 | 0.03 | 0.08 |
| nonrefusal | gated sens0.5 | knownw | lag | 0.97 | 0.00 | 0.65 | 0.545 | 0.015 | 0.523 | 0.908 | 1.7 | 0.01 | 0.00 |
| nonrefusal | gated sens0.5 | knownw | lag_floor | 1.00 | 0.00 | 0.20 | 0.541 | 0.018 | 0.520 | 0.943 | 5.4 | 0.00 | 0.11 |
| nonrefusal | gated sens0.5 | est30w | lag | 0.90 | 0.00 | 0.32 | 0.526 | 0.020 | 0.503 | 0.938 | 8.7 | 0.00 | 0.00 |
| nonrefusal | gated sens0.5 | est30w | lag_floor | 0.91 | 0.00 | 0.20 | 0.529 | 0.020 | 0.506 | 0.943 | 10.9 | 0.00 | 0.02 |
| nonrefusal | gated sens0.5 | est100w | lag | 0.97 | 0.00 | 0.38 | 0.535 | 0.017 | 0.512 | 0.933 | 4.7 | 0.00 | 0.00 |
| nonrefusal | gated sens0.5 | est100w | lag_floor | 0.97 | 0.00 | 0.20 | 0.534 | 0.019 | 0.511 | 0.942 | 7.6 | 0.00 | 0.02 |
| nonrefusal | gated sens0.8 | naive | lag | 0.98 | 0.00 | 0.76 | 0.544 | 0.014 | 0.520 | 0.900 | 0.7 | 0.48 | 0.08 |
| nonrefusal | gated sens0.8 | naive | lag_floor | 1.00 | 0.00 | 0.09 | 0.531 | 0.022 | 0.501 | 0.959 | 5.0 | 0.25 | 0.78 |
| nonrefusal | gated sens0.8 | known | lag | 0.99 | 0.00 | 0.72 | 0.545 | 0.015 | 0.524 | 0.915 | 1.0 | 0.58 | 0.06 |
| nonrefusal | gated sens0.8 | known | lag_floor | 1.00 | 0.00 | 0.09 | 0.531 | 0.022 | 0.501 | 0.959 | 5.0 | 0.67 | 0.63 |
| nonrefusal | gated sens0.8 | est3 | lag | 0.43 | 0.00 | 0.09 | 0.526 | 0.024 | 0.495 | 0.962 | 16.3 | 0.01 | 0.00 |
| nonrefusal | gated sens0.8 | est3 | lag_floor | 0.44 | 0.00 | 0.09 | 0.525 | 0.024 | 0.493 | 0.962 | 16.9 | 0.00 | 0.00 |
| nonrefusal | gated sens0.8 | est30 | lag | 0.98 | 0.00 | 0.34 | 0.534 | 0.020 | 0.504 | 0.942 | 2.2 | 0.04 | 0.00 |
| nonrefusal | gated sens0.8 | est30 | lag_floor | 1.00 | 0.00 | 0.09 | 0.531 | 0.022 | 0.501 | 0.960 | 5.3 | 0.03 | 0.24 |
| nonrefusal | gated sens0.8 | est100 | lag | 0.97 | 0.00 | 0.43 | 0.539 | 0.017 | 0.514 | 0.930 | 1.6 | 0.06 | 0.02 |
| nonrefusal | gated sens0.8 | est100 | lag_floor | 1.00 | 0.00 | 0.09 | 0.531 | 0.022 | 0.501 | 0.959 | 5.1 | 0.06 | 0.34 |
| nonrefusal | gated sens0.8 | knownw | lag | 0.99 | 0.00 | 0.72 | 0.545 | 0.015 | 0.524 | 0.915 | 1.0 | 0.02 | 0.03 |
| nonrefusal | gated sens0.8 | knownw | lag_floor | 1.00 | 0.00 | 0.09 | 0.531 | 0.022 | 0.501 | 0.959 | 5.0 | 0.01 | 0.50 |
| nonrefusal | gated sens0.8 | est30w | lag | 0.98 | 0.00 | 0.34 | 0.534 | 0.020 | 0.504 | 0.942 | 2.2 | 0.00 | 0.00 |
| nonrefusal | gated sens0.8 | est30w | lag_floor | 1.00 | 0.00 | 0.09 | 0.531 | 0.022 | 0.501 | 0.960 | 5.3 | 0.00 | 0.16 |
| nonrefusal | gated sens0.8 | est100w | lag | 0.97 | 0.00 | 0.43 | 0.539 | 0.017 | 0.514 | 0.930 | 1.6 | 0.00 | 0.00 |
| nonrefusal | gated sens0.8 | est100w | lag_floor | 1.00 | 0.00 | 0.09 | 0.531 | 0.022 | 0.501 | 0.959 | 5.1 | 0.00 | 0.30 |
| nonrefusal | gated sens1 | naive | lag | 0.94 | 0.00 | 0.66 | 0.538 | 0.017 | 0.513 | 0.920 | 0.9 | 0.04 | 0.01 |
| nonrefusal | gated sens1 | naive | lag_floor | 1.00 | 0.00 | 0.06 | 0.526 | 0.023 | 0.497 | 0.968 | 5.0 | 0.03 | 0.57 |
| nonrefusal | gated sens1 | known | lag | 0.98 | 0.00 | 0.71 | 0.542 | 0.014 | 0.520 | 0.910 | 0.8 | 0.49 | 0.07 |
| nonrefusal | gated sens1 | known | lag_floor | 1.00 | 0.00 | 0.06 | 0.526 | 0.023 | 0.497 | 0.968 | 5.0 | 0.66 | 0.76 |
| nonrefusal | gated sens1 | est3 | lag | 0.88 | 0.00 | 0.06 | 0.498 | 0.028 | 0.470 | 0.966 | 13.9 | 0.02 | 0.00 |
| nonrefusal | gated sens1 | est3 | lag_floor | 0.88 | 0.00 | 0.06 | 0.498 | 0.028 | 0.469 | 0.966 | 14.7 | 0.02 | 0.00 |
| nonrefusal | gated sens1 | est30 | lag | 0.99 | 0.00 | 0.59 | 0.535 | 0.020 | 0.505 | 0.934 | 1.5 | 0.06 | 0.01 |
| nonrefusal | gated sens1 | est30 | lag_floor | 1.00 | 0.00 | 0.06 | 0.526 | 0.023 | 0.497 | 0.968 | 5.0 | 0.10 | 0.44 |
| nonrefusal | gated sens1 | est100 | lag | 0.96 | 0.00 | 0.62 | 0.537 | 0.017 | 0.511 | 0.923 | 0.8 | 0.11 | 0.03 |
| nonrefusal | gated sens1 | est100 | lag_floor | 1.00 | 0.00 | 0.06 | 0.526 | 0.023 | 0.497 | 0.968 | 5.0 | 0.11 | 0.53 |
| nonrefusal | gated sens1 | knownw | lag | 0.98 | 0.00 | 0.71 | 0.542 | 0.014 | 0.520 | 0.910 | 0.8 | 0.03 | 0.01 |
| nonrefusal | gated sens1 | knownw | lag_floor | 1.00 | 0.00 | 0.06 | 0.526 | 0.023 | 0.497 | 0.968 | 5.0 | 0.03 | 0.56 |
| nonrefusal | gated sens1 | est30w | lag | 0.99 | 0.00 | 0.59 | 0.535 | 0.020 | 0.505 | 0.934 | 1.5 | 0.01 | 0.00 |
| nonrefusal | gated sens1 | est30w | lag_floor | 1.00 | 0.00 | 0.06 | 0.526 | 0.023 | 0.497 | 0.968 | 5.0 | 0.00 | 0.38 |
| nonrefusal | gated sens1 | est100w | lag | 0.96 | 0.00 | 0.62 | 0.537 | 0.017 | 0.511 | 0.923 | 0.8 | 0.01 | 0.01 |
| nonrefusal | gated sens1 | est100w | lag_floor | 1.00 | 0.00 | 0.06 | 0.526 | 0.023 | 0.497 | 0.968 | 5.0 | 0.03 | 0.48 |
