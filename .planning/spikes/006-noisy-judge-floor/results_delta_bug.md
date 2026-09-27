# Spike 006 results

tau 0.12, checks every 25, delta 0.1, calibration delta 0.05 (200 negatives, N positives). Shares over seeds; `ret. viol.` = a returned policy above tau (the Seldonian failure); `cert miss` = the trajectory bound below the true rate at some check; `clear` = the bound claims every check below tau.

## noise model: hash

| noise | judge | mode | method | solution | ret. viol. | entered U | task acc | task refusal | twin acc | F refusal | lam end | cert miss | clear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hash | exact | naive | lag | 0.95 | 0.00 | 0.70 | 0.538 | 0.019 | 0.506 | 0.921 | 1.7 | 0.13 | 0.04 |
| hash | exact | naive | lag_floor | 1.00 | 0.00 | 0.16 | 0.528 | 0.022 | 0.503 | 0.968 | 5.0 | 0.07 | 0.69 |
| hash | ungated sens0.5 | naive | lag | 0.00 | 0.00 | 0.91 | 0.532 | 0.016 | 0.497 | 0.616 | 20.0 | 0.06 | 0.00 |
| hash | ungated sens0.5 | naive | lag_floor | 0.00 | 0.00 | 0.91 | 0.532 | 0.016 | 0.497 | 0.616 | 20.0 | 0.06 | 0.00 |
| hash | ungated sens0.5 | known | lag | 0.22 | 0.02 | 0.89 | 0.481 | 0.020 | 0.447 | 0.610 | 18.3 | 0.21 | 0.00 |
| hash | ungated sens0.5 | known | lag_floor | 0.21 | 0.03 | 0.91 | 0.480 | 0.021 | 0.446 | 0.612 | 18.8 | 0.21 | 0.00 |
| hash | ungated sens0.5 | est3 | lag | 0.00 | 0.00 | 0.91 | 0.532 | 0.016 | 0.497 | 0.616 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens0.5 | est3 | lag_floor | 0.00 | 0.00 | 0.91 | 0.532 | 0.016 | 0.497 | 0.616 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens0.5 | est30 | lag | 0.00 | 0.00 | 0.91 | 0.530 | 0.016 | 0.495 | 0.614 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens0.5 | est30 | lag_floor | 0.00 | 0.00 | 0.91 | 0.530 | 0.016 | 0.495 | 0.614 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens0.5 | est100 | lag | 0.01 | 0.00 | 0.91 | 0.526 | 0.016 | 0.492 | 0.610 | 20.0 | 0.01 | 0.00 |
| hash | ungated sens0.5 | est100 | lag_floor | 0.01 | 0.00 | 0.91 | 0.526 | 0.016 | 0.492 | 0.610 | 20.0 | 0.01 | 0.00 |
| hash | ungated sens0.8 | naive | lag | 0.00 | 0.00 | 0.53 | 0.527 | 0.024 | 0.488 | 0.814 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens0.8 | naive | lag_floor | 0.00 | 0.00 | 0.53 | 0.527 | 0.024 | 0.488 | 0.814 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens0.8 | known | lag | 0.65 | 0.00 | 0.62 | 0.495 | 0.025 | 0.460 | 0.810 | 11.8 | 0.25 | 0.00 |
| hash | ungated sens0.8 | known | lag_floor | 0.67 | 0.00 | 0.53 | 0.483 | 0.027 | 0.449 | 0.799 | 13.7 | 0.23 | 0.00 |
| hash | ungated sens0.8 | est3 | lag | 0.00 | 0.00 | 0.53 | 0.527 | 0.024 | 0.488 | 0.814 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens0.8 | est3 | lag_floor | 0.00 | 0.00 | 0.53 | 0.527 | 0.024 | 0.488 | 0.814 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens0.8 | est30 | lag | 0.05 | 0.00 | 0.53 | 0.519 | 0.025 | 0.481 | 0.813 | 19.7 | 0.02 | 0.00 |
| hash | ungated sens0.8 | est30 | lag_floor | 0.05 | 0.00 | 0.53 | 0.519 | 0.025 | 0.481 | 0.813 | 19.7 | 0.02 | 0.00 |
| hash | ungated sens0.8 | est100 | lag | 0.11 | 0.00 | 0.54 | 0.508 | 0.026 | 0.470 | 0.809 | 19.4 | 0.02 | 0.00 |
| hash | ungated sens0.8 | est100 | lag_floor | 0.11 | 0.00 | 0.53 | 0.508 | 0.026 | 0.470 | 0.809 | 19.5 | 0.02 | 0.00 |
| hash | ungated sens1 | naive | lag | 0.00 | 0.00 | 0.24 | 0.522 | 0.026 | 0.487 | 0.886 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens1 | naive | lag_floor | 0.00 | 0.00 | 0.24 | 0.522 | 0.026 | 0.487 | 0.886 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens1 | known | lag | 0.92 | 0.00 | 0.52 | 0.509 | 0.025 | 0.474 | 0.879 | 6.4 | 0.23 | 0.01 |
| hash | ungated sens1 | known | lag_floor | 0.95 | 0.00 | 0.24 | 0.498 | 0.029 | 0.461 | 0.877 | 9.5 | 0.27 | 0.03 |
| hash | ungated sens1 | est3 | lag | 0.01 | 0.00 | 0.24 | 0.520 | 0.027 | 0.486 | 0.886 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens1 | est3 | lag_floor | 0.01 | 0.00 | 0.24 | 0.520 | 0.027 | 0.486 | 0.886 | 20.0 | 0.00 | 0.00 |
| hash | ungated sens1 | est30 | lag | 0.27 | 0.00 | 0.29 | 0.505 | 0.027 | 0.469 | 0.875 | 18.0 | 0.01 | 0.00 |
| hash | ungated sens1 | est30 | lag_floor | 0.29 | 0.00 | 0.24 | 0.501 | 0.028 | 0.465 | 0.875 | 18.1 | 0.02 | 0.00 |
| hash | ungated sens1 | est100 | lag | 0.33 | 0.00 | 0.26 | 0.500 | 0.028 | 0.463 | 0.872 | 16.9 | 0.02 | 0.01 |
| hash | ungated sens1 | est100 | lag_floor | 0.35 | 0.00 | 0.24 | 0.497 | 0.028 | 0.460 | 0.874 | 17.3 | 0.02 | 0.01 |
| hash | gated sens0.5 | naive | lag | 1.00 | 0.16 | 0.80 | 0.553 | 0.008 | 0.541 | 0.833 | 0.8 | 1.00 | 0.37 |
| hash | gated sens0.5 | naive | lag_floor | 1.00 | 0.01 | 0.34 | 0.540 | 0.018 | 0.512 | 0.905 | 5.1 | 0.89 | 0.85 |
| hash | gated sens0.5 | known | lag | 0.95 | 0.00 | 0.63 | 0.532 | 0.018 | 0.502 | 0.902 | 4.7 | 0.11 | 0.03 |
| hash | gated sens0.5 | known | lag_floor | 0.95 | 0.00 | 0.34 | 0.530 | 0.021 | 0.499 | 0.906 | 7.9 | 0.09 | 0.10 |
| hash | gated sens0.5 | est3 | lag | 0.13 | 0.00 | 0.34 | 0.544 | 0.019 | 0.515 | 0.914 | 18.6 | 0.00 | 0.00 |
| hash | gated sens0.5 | est3 | lag_floor | 0.14 | 0.00 | 0.34 | 0.543 | 0.019 | 0.515 | 0.914 | 19.0 | 0.00 | 0.00 |
| hash | gated sens0.5 | est30 | lag | 0.48 | 0.00 | 0.38 | 0.519 | 0.021 | 0.487 | 0.908 | 14.4 | 0.00 | 0.00 |
| hash | gated sens0.5 | est30 | lag_floor | 0.46 | 0.00 | 0.34 | 0.512 | 0.022 | 0.480 | 0.903 | 16.1 | 0.00 | 0.00 |
| hash | gated sens0.5 | est100 | lag | 0.73 | 0.00 | 0.43 | 0.511 | 0.021 | 0.482 | 0.899 | 11.7 | 0.01 | 0.00 |
| hash | gated sens0.5 | est100 | lag_floor | 0.71 | 0.00 | 0.34 | 0.509 | 0.022 | 0.481 | 0.897 | 13.9 | 0.00 | 0.01 |
| hash | gated sens0.8 | naive | lag | 0.98 | 0.00 | 0.71 | 0.540 | 0.016 | 0.515 | 0.915 | 1.2 | 0.18 | 0.09 |
| hash | gated sens0.8 | naive | lag_floor | 1.00 | 0.00 | 0.09 | 0.532 | 0.023 | 0.499 | 0.953 | 5.0 | 0.02 | 0.72 |
| hash | gated sens0.8 | known | lag | 0.98 | 0.00 | 0.64 | 0.537 | 0.017 | 0.509 | 0.917 | 1.4 | 0.06 | 0.05 |
| hash | gated sens0.8 | known | lag_floor | 1.00 | 0.00 | 0.09 | 0.532 | 0.023 | 0.499 | 0.953 | 5.1 | 0.10 | 0.51 |
| hash | gated sens0.8 | est3 | lag | 0.21 | 0.00 | 0.09 | 0.523 | 0.025 | 0.488 | 0.953 | 18.2 | 0.00 | 0.00 |
| hash | gated sens0.8 | est3 | lag_floor | 0.22 | 0.00 | 0.09 | 0.524 | 0.025 | 0.488 | 0.953 | 18.6 | 0.00 | 0.00 |
| hash | gated sens0.8 | est30 | lag | 0.98 | 0.00 | 0.22 | 0.526 | 0.022 | 0.496 | 0.944 | 5.2 | 0.00 | 0.01 |
| hash | gated sens0.8 | est30 | lag_floor | 0.99 | 0.00 | 0.09 | 0.528 | 0.023 | 0.495 | 0.955 | 7.8 | 0.00 | 0.12 |
| hash | gated sens0.8 | est100 | lag | 0.98 | 0.00 | 0.31 | 0.534 | 0.020 | 0.505 | 0.940 | 2.7 | 0.01 | 0.00 |
| hash | gated sens0.8 | est100 | lag_floor | 1.00 | 0.00 | 0.09 | 0.529 | 0.023 | 0.496 | 0.955 | 5.6 | 0.00 | 0.23 |
| hash | gated sens1 | naive | lag | 0.98 | 0.00 | 0.48 | 0.534 | 0.019 | 0.508 | 0.937 | 1.6 | 0.01 | 0.02 |
| hash | gated sens1 | naive | lag_floor | 1.00 | 0.00 | 0.06 | 0.524 | 0.024 | 0.496 | 0.965 | 5.0 | 0.00 | 0.48 |
| hash | gated sens1 | known | lag | 0.96 | 0.00 | 0.66 | 0.539 | 0.017 | 0.513 | 0.918 | 1.1 | 0.07 | 0.01 |
| hash | gated sens1 | known | lag_floor | 1.00 | 0.00 | 0.06 | 0.524 | 0.024 | 0.496 | 0.965 | 5.0 | 0.07 | 0.63 |
| hash | gated sens1 | est3 | lag | 0.56 | 0.00 | 0.06 | 0.495 | 0.029 | 0.462 | 0.962 | 17.1 | 0.00 | 0.00 |
| hash | gated sens1 | est3 | lag_floor | 0.56 | 0.00 | 0.06 | 0.494 | 0.029 | 0.461 | 0.962 | 17.5 | 0.00 | 0.00 |
| hash | gated sens1 | est30 | lag | 0.98 | 0.00 | 0.38 | 0.529 | 0.021 | 0.500 | 0.939 | 1.9 | 0.01 | 0.00 |
| hash | gated sens1 | est30 | lag_floor | 1.00 | 0.00 | 0.06 | 0.524 | 0.024 | 0.495 | 0.965 | 5.1 | 0.00 | 0.38 |
| hash | gated sens1 | est100 | lag | 1.00 | 0.00 | 0.50 | 0.535 | 0.020 | 0.510 | 0.938 | 1.4 | 0.02 | 0.02 |
| hash | gated sens1 | est100 | lag_floor | 1.00 | 0.00 | 0.06 | 0.524 | 0.024 | 0.496 | 0.965 | 5.0 | 0.00 | 0.46 |

## noise model: nonrefusal

| noise | judge | mode | method | solution | ret. viol. | entered U | task acc | task refusal | twin acc | F refusal | lam end | cert miss | clear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nonrefusal | exact | naive | lag | 0.95 | 0.00 | 0.70 | 0.538 | 0.019 | 0.506 | 0.921 | 1.7 | 0.13 | 0.04 |
| nonrefusal | exact | naive | lag_floor | 1.00 | 0.00 | 0.16 | 0.528 | 0.022 | 0.503 | 0.968 | 5.0 | 0.07 | 0.69 |
| nonrefusal | ungated sens0.5 | naive | lag | 0.97 | 0.19 | 0.87 | 0.549 | 0.014 | 0.533 | 0.878 | 0.8 | 0.98 | 0.07 |
| nonrefusal | ungated sens0.5 | naive | lag_floor | 1.00 | 0.00 | 0.08 | 0.542 | 0.022 | 0.519 | 0.954 | 5.0 | 0.88 | 0.28 |
| nonrefusal | ungated sens0.5 | known | lag | 1.00 | 0.25 | 0.79 | 0.564 | 0.004 | 0.562 | 0.755 | 0.0 | 1.00 | 0.90 |
| nonrefusal | ungated sens0.5 | known | lag_floor | 1.00 | 0.00 | 0.08 | 0.542 | 0.022 | 0.519 | 0.954 | 5.0 | 1.00 | 1.00 |
| nonrefusal | ungated sens0.5 | est3 | lag | 1.00 | 0.28 | 0.84 | 0.552 | 0.012 | 0.537 | 0.844 | 0.4 | 0.09 | 0.05 |
| nonrefusal | ungated sens0.5 | est3 | lag_floor | 1.00 | 0.00 | 0.08 | 0.542 | 0.022 | 0.519 | 0.954 | 5.0 | 0.09 | 0.07 |
| nonrefusal | ungated sens0.5 | est30 | lag | 1.00 | 0.25 | 0.80 | 0.557 | 0.009 | 0.546 | 0.814 | 0.4 | 1.00 | 0.53 |
| nonrefusal | ungated sens0.5 | est30 | lag_floor | 1.00 | 0.00 | 0.08 | 0.542 | 0.022 | 0.519 | 0.954 | 5.0 | 1.00 | 0.79 |
| nonrefusal | ungated sens0.5 | est100 | lag | 1.00 | 0.30 | 0.80 | 0.558 | 0.008 | 0.548 | 0.802 | 0.2 | 1.00 | 0.59 |
| nonrefusal | ungated sens0.5 | est100 | lag_floor | 1.00 | 0.00 | 0.08 | 0.542 | 0.022 | 0.519 | 0.954 | 5.0 | 1.00 | 0.78 |
| nonrefusal | ungated sens0.5 | knownw | lag | 1.00 | 0.25 | 0.79 | 0.564 | 0.004 | 0.562 | 0.755 | 0.0 | 0.00 | 0.00 |
| nonrefusal | ungated sens0.5 | knownw | lag_floor | 1.00 | 0.00 | 0.08 | 0.542 | 0.022 | 0.519 | 0.954 | 5.0 | 0.00 | 0.00 |
| nonrefusal | ungated sens0.5 | est30w | lag | 1.00 | 0.25 | 0.80 | 0.557 | 0.009 | 0.546 | 0.814 | 0.4 | 0.00 | 0.00 |
| nonrefusal | ungated sens0.5 | est30w | lag_floor | 1.00 | 0.00 | 0.08 | 0.542 | 0.022 | 0.519 | 0.954 | 5.0 | 0.00 | 0.00 |
| nonrefusal | ungated sens0.5 | est100w | lag | 1.00 | 0.30 | 0.80 | 0.558 | 0.008 | 0.548 | 0.802 | 0.2 | 0.00 | 0.00 |
| nonrefusal | ungated sens0.5 | est100w | lag_floor | 1.00 | 0.00 | 0.08 | 0.542 | 0.022 | 0.519 | 0.954 | 5.0 | 0.00 | 0.00 |
| nonrefusal | ungated sens0.8 | naive | lag | 0.99 | 0.00 | 0.76 | 0.539 | 0.021 | 0.516 | 0.927 | 1.5 | 0.15 | 0.01 |
| nonrefusal | ungated sens0.8 | naive | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.025 | 0.512 | 0.966 | 5.0 | 0.16 | 0.16 |
| nonrefusal | ungated sens0.8 | known | lag | 1.00 | 0.23 | 0.79 | 0.562 | 0.005 | 0.557 | 0.770 | 0.1 | 1.00 | 0.79 |
| nonrefusal | ungated sens0.8 | known | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.025 | 0.512 | 0.966 | 5.0 | 1.00 | 0.99 |
| nonrefusal | ungated sens0.8 | est3 | lag | 1.00 | 0.08 | 0.80 | 0.549 | 0.013 | 0.530 | 0.876 | 0.3 | 0.49 | 0.17 |
| nonrefusal | ungated sens0.8 | est3 | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.025 | 0.512 | 0.966 | 5.0 | 0.49 | 0.27 |
| nonrefusal | ungated sens0.8 | est30 | lag | 1.00 | 0.24 | 0.82 | 0.555 | 0.010 | 0.536 | 0.836 | 0.4 | 1.00 | 0.53 |
| nonrefusal | ungated sens0.8 | est30 | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.025 | 0.512 | 0.966 | 5.0 | 1.00 | 0.81 |
| nonrefusal | ungated sens0.8 | est100 | lag | 0.99 | 0.25 | 0.80 | 0.556 | 0.010 | 0.536 | 0.829 | 0.3 | 1.00 | 0.57 |
| nonrefusal | ungated sens0.8 | est100 | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.025 | 0.512 | 0.966 | 5.0 | 1.00 | 0.82 |
| nonrefusal | ungated sens0.8 | knownw | lag | 1.00 | 0.23 | 0.79 | 0.562 | 0.005 | 0.557 | 0.770 | 0.1 | 0.01 | 0.00 |
| nonrefusal | ungated sens0.8 | knownw | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.025 | 0.512 | 0.966 | 5.0 | 0.01 | 0.11 |
| nonrefusal | ungated sens0.8 | est30w | lag | 1.00 | 0.24 | 0.82 | 0.555 | 0.010 | 0.536 | 0.836 | 0.4 | 0.01 | 0.00 |
| nonrefusal | ungated sens0.8 | est30w | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.025 | 0.512 | 0.966 | 5.0 | 0.01 | 0.02 |
| nonrefusal | ungated sens0.8 | est100w | lag | 0.99 | 0.25 | 0.80 | 0.556 | 0.010 | 0.536 | 0.829 | 0.3 | 0.01 | 0.00 |
| nonrefusal | ungated sens0.8 | est100w | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.025 | 0.512 | 0.966 | 5.0 | 0.01 | 0.04 |
| nonrefusal | ungated sens1 | naive | lag | 0.97 | 0.00 | 0.48 | 0.538 | 0.022 | 0.511 | 0.938 | 1.3 | 0.03 | 0.00 |
| nonrefusal | ungated sens1 | naive | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.026 | 0.506 | 0.972 | 5.0 | 0.06 | 0.10 |
| nonrefusal | ungated sens1 | known | lag | 1.00 | 0.22 | 0.79 | 0.560 | 0.006 | 0.553 | 0.787 | 0.1 | 1.00 | 0.75 |
| nonrefusal | ungated sens1 | known | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.026 | 0.506 | 0.972 | 5.0 | 1.00 | 0.99 |
| nonrefusal | ungated sens1 | est3 | lag | 0.98 | 0.09 | 0.79 | 0.550 | 0.016 | 0.522 | 0.882 | 0.9 | 1.00 | 0.22 |
| nonrefusal | ungated sens1 | est3 | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.026 | 0.506 | 0.972 | 5.0 | 1.00 | 0.49 |
| nonrefusal | ungated sens1 | est30 | lag | 0.98 | 0.23 | 0.80 | 0.553 | 0.009 | 0.536 | 0.828 | 0.3 | 1.00 | 0.53 |
| nonrefusal | ungated sens1 | est30 | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.026 | 0.506 | 0.972 | 5.0 | 1.00 | 0.82 |
| nonrefusal | ungated sens1 | est100 | lag | 0.99 | 0.20 | 0.80 | 0.553 | 0.009 | 0.534 | 0.835 | 0.4 | 1.00 | 0.57 |
| nonrefusal | ungated sens1 | est100 | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.026 | 0.506 | 0.972 | 5.0 | 1.00 | 0.86 |
| nonrefusal | ungated sens1 | knownw | lag | 1.00 | 0.22 | 0.79 | 0.560 | 0.006 | 0.553 | 0.787 | 0.1 | 0.00 | 0.00 |
| nonrefusal | ungated sens1 | knownw | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.026 | 0.506 | 0.972 | 5.0 | 0.01 | 0.35 |
| nonrefusal | ungated sens1 | est30w | lag | 0.98 | 0.23 | 0.80 | 0.553 | 0.009 | 0.536 | 0.828 | 0.3 | 0.00 | 0.00 |
| nonrefusal | ungated sens1 | est30w | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.026 | 0.506 | 0.972 | 5.0 | 0.00 | 0.10 |
| nonrefusal | ungated sens1 | est100w | lag | 0.99 | 0.20 | 0.80 | 0.553 | 0.009 | 0.534 | 0.835 | 0.4 | 0.00 | 0.00 |
| nonrefusal | ungated sens1 | est100w | lag_floor | 1.00 | 0.00 | 0.06 | 0.535 | 0.026 | 0.506 | 0.972 | 5.0 | 0.01 | 0.18 |
| nonrefusal | gated sens0.5 | naive | lag | 0.98 | 0.24 | 0.79 | 0.558 | 0.006 | 0.552 | 0.814 | 0.5 | 1.00 | 0.47 |
| nonrefusal | gated sens0.5 | naive | lag_floor | 1.00 | 0.00 | 0.20 | 0.542 | 0.017 | 0.522 | 0.941 | 5.0 | 1.00 | 0.95 |
| nonrefusal | gated sens0.5 | known | lag | 0.97 | 0.00 | 0.65 | 0.545 | 0.015 | 0.523 | 0.908 | 1.7 | 0.78 | 0.07 |
| nonrefusal | gated sens0.5 | known | lag_floor | 1.00 | 0.00 | 0.20 | 0.541 | 0.018 | 0.520 | 0.943 | 5.4 | 0.90 | 0.54 |
| nonrefusal | gated sens0.5 | est3 | lag | 0.32 | 0.00 | 0.25 | 0.542 | 0.018 | 0.519 | 0.946 | 15.8 | 0.01 | 0.00 |
| nonrefusal | gated sens0.5 | est3 | lag_floor | 0.32 | 0.00 | 0.20 | 0.540 | 0.019 | 0.517 | 0.947 | 16.7 | 0.02 | 0.01 |
| nonrefusal | gated sens0.5 | est30 | lag | 0.93 | 0.00 | 0.35 | 0.530 | 0.018 | 0.507 | 0.934 | 7.1 | 0.09 | 0.03 |
| nonrefusal | gated sens0.5 | est30 | lag_floor | 0.96 | 0.00 | 0.20 | 0.528 | 0.020 | 0.504 | 0.942 | 9.7 | 0.11 | 0.09 |
| nonrefusal | gated sens0.5 | est100 | lag | 0.96 | 0.00 | 0.41 | 0.535 | 0.017 | 0.511 | 0.932 | 4.1 | 0.17 | 0.02 |
| nonrefusal | gated sens0.5 | est100 | lag_floor | 0.98 | 0.00 | 0.20 | 0.536 | 0.018 | 0.514 | 0.943 | 7.2 | 0.16 | 0.13 |
| nonrefusal | gated sens0.5 | knownw | lag | 0.97 | 0.00 | 0.65 | 0.545 | 0.015 | 0.523 | 0.908 | 1.7 | 0.03 | 0.00 |
| nonrefusal | gated sens0.5 | knownw | lag_floor | 1.00 | 0.00 | 0.20 | 0.541 | 0.018 | 0.520 | 0.943 | 5.4 | 0.03 | 0.17 |
| nonrefusal | gated sens0.5 | est30w | lag | 0.93 | 0.00 | 0.35 | 0.530 | 0.018 | 0.507 | 0.934 | 7.1 | 0.00 | 0.00 |
| nonrefusal | gated sens0.5 | est30w | lag_floor | 0.96 | 0.00 | 0.20 | 0.528 | 0.020 | 0.504 | 0.942 | 9.7 | 0.00 | 0.03 |
| nonrefusal | gated sens0.5 | est100w | lag | 0.96 | 0.00 | 0.41 | 0.535 | 0.017 | 0.511 | 0.932 | 4.1 | 0.00 | 0.00 |
| nonrefusal | gated sens0.5 | est100w | lag_floor | 0.98 | 0.00 | 0.20 | 0.536 | 0.018 | 0.514 | 0.943 | 7.2 | 0.01 | 0.05 |
| nonrefusal | gated sens0.8 | naive | lag | 0.98 | 0.00 | 0.76 | 0.544 | 0.014 | 0.520 | 0.900 | 0.7 | 0.68 | 0.13 |
| nonrefusal | gated sens0.8 | naive | lag_floor | 1.00 | 0.00 | 0.09 | 0.531 | 0.022 | 0.501 | 0.959 | 5.0 | 0.41 | 0.80 |
| nonrefusal | gated sens0.8 | known | lag | 0.99 | 0.00 | 0.72 | 0.545 | 0.015 | 0.524 | 0.915 | 1.0 | 0.69 | 0.07 |
| nonrefusal | gated sens0.8 | known | lag_floor | 1.00 | 0.00 | 0.09 | 0.531 | 0.022 | 0.501 | 0.959 | 5.0 | 0.80 | 0.71 |
| nonrefusal | gated sens0.8 | est3 | lag | 0.49 | 0.00 | 0.09 | 0.533 | 0.022 | 0.503 | 0.961 | 14.3 | 0.01 | 0.00 |
| nonrefusal | gated sens0.8 | est3 | lag_floor | 0.48 | 0.00 | 0.09 | 0.532 | 0.023 | 0.501 | 0.961 | 15.2 | 0.01 | 0.01 |
| nonrefusal | gated sens0.8 | est30 | lag | 0.97 | 0.00 | 0.34 | 0.538 | 0.019 | 0.509 | 0.938 | 2.2 | 0.06 | 0.01 |
| nonrefusal | gated sens0.8 | est30 | lag_floor | 1.00 | 0.00 | 0.09 | 0.531 | 0.022 | 0.501 | 0.959 | 5.2 | 0.14 | 0.31 |
| nonrefusal | gated sens0.8 | est100 | lag | 0.97 | 0.00 | 0.44 | 0.538 | 0.018 | 0.514 | 0.927 | 1.4 | 0.14 | 0.05 |
| nonrefusal | gated sens0.8 | est100 | lag_floor | 1.00 | 0.00 | 0.09 | 0.531 | 0.022 | 0.501 | 0.959 | 5.0 | 0.19 | 0.42 |
| nonrefusal | gated sens0.8 | knownw | lag | 0.99 | 0.00 | 0.72 | 0.545 | 0.015 | 0.524 | 0.915 | 1.0 | 0.07 | 0.03 |
| nonrefusal | gated sens0.8 | knownw | lag_floor | 1.00 | 0.00 | 0.09 | 0.531 | 0.022 | 0.501 | 0.959 | 5.0 | 0.04 | 0.56 |
| nonrefusal | gated sens0.8 | est30w | lag | 0.97 | 0.00 | 0.34 | 0.538 | 0.019 | 0.509 | 0.938 | 2.2 | 0.00 | 0.00 |
| nonrefusal | gated sens0.8 | est30w | lag_floor | 1.00 | 0.00 | 0.09 | 0.531 | 0.022 | 0.501 | 0.959 | 5.2 | 0.00 | 0.25 |
| nonrefusal | gated sens0.8 | est100w | lag | 0.97 | 0.00 | 0.44 | 0.538 | 0.018 | 0.514 | 0.927 | 1.4 | 0.00 | 0.01 |
| nonrefusal | gated sens0.8 | est100w | lag_floor | 1.00 | 0.00 | 0.09 | 0.531 | 0.022 | 0.501 | 0.959 | 5.0 | 0.01 | 0.35 |
| nonrefusal | gated sens1 | naive | lag | 0.94 | 0.00 | 0.66 | 0.538 | 0.017 | 0.513 | 0.920 | 0.9 | 0.08 | 0.04 |
| nonrefusal | gated sens1 | naive | lag_floor | 1.00 | 0.00 | 0.06 | 0.526 | 0.023 | 0.497 | 0.968 | 5.0 | 0.09 | 0.63 |
| nonrefusal | gated sens1 | known | lag | 0.98 | 0.00 | 0.71 | 0.542 | 0.014 | 0.520 | 0.910 | 0.8 | 0.64 | 0.09 |
| nonrefusal | gated sens1 | known | lag_floor | 1.00 | 0.00 | 0.06 | 0.526 | 0.023 | 0.497 | 0.968 | 5.0 | 0.79 | 0.76 |
| nonrefusal | gated sens1 | est3 | lag | 0.95 | 0.00 | 0.06 | 0.511 | 0.027 | 0.481 | 0.968 | 9.4 | 0.04 | 0.00 |
| nonrefusal | gated sens1 | est3 | lag_floor | 0.95 | 0.00 | 0.06 | 0.516 | 0.026 | 0.486 | 0.970 | 10.8 | 0.04 | 0.00 |
| nonrefusal | gated sens1 | est30 | lag | 0.99 | 0.00 | 0.62 | 0.535 | 0.019 | 0.505 | 0.929 | 1.5 | 0.13 | 0.02 |
| nonrefusal | gated sens1 | est30 | lag_floor | 1.00 | 0.00 | 0.06 | 0.526 | 0.023 | 0.497 | 0.968 | 5.0 | 0.19 | 0.51 |
| nonrefusal | gated sens1 | est100 | lag | 0.95 | 0.00 | 0.64 | 0.539 | 0.016 | 0.515 | 0.920 | 0.7 | 0.19 | 0.04 |
| nonrefusal | gated sens1 | est100 | lag_floor | 1.00 | 0.00 | 0.06 | 0.526 | 0.023 | 0.497 | 0.968 | 5.0 | 0.25 | 0.60 |
| nonrefusal | gated sens1 | knownw | lag | 0.98 | 0.00 | 0.71 | 0.542 | 0.014 | 0.520 | 0.910 | 0.8 | 0.06 | 0.03 |
| nonrefusal | gated sens1 | knownw | lag_floor | 1.00 | 0.00 | 0.06 | 0.526 | 0.023 | 0.497 | 0.968 | 5.0 | 0.03 | 0.62 |
| nonrefusal | gated sens1 | est30w | lag | 0.99 | 0.00 | 0.62 | 0.535 | 0.019 | 0.505 | 0.929 | 1.5 | 0.02 | 0.01 |
| nonrefusal | gated sens1 | est30w | lag_floor | 1.00 | 0.00 | 0.06 | 0.526 | 0.023 | 0.497 | 0.968 | 5.0 | 0.03 | 0.43 |
| nonrefusal | gated sens1 | est100w | lag | 0.95 | 0.00 | 0.64 | 0.539 | 0.016 | 0.515 | 0.920 | 0.7 | 0.04 | 0.02 |
| nonrefusal | gated sens1 | est100w | lag_floor | 1.00 | 0.00 | 0.06 | 0.526 | 0.023 | 0.497 | 0.968 | 5.0 | 0.03 | 0.52 |
