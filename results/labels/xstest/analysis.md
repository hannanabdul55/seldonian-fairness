# Qwen3Guard-4B's refusal flag against XSTest's human labels

2250 completions from 5 model variants; 0 without a Refusal line in the guard's verdict. Intervals are 95%: Clopper-Pearson per model, a bootstrap over prompts for the pooled rows.

## Safe prompts (250 per model): the over-refusal setting

### Human event: full refusal (strict)

| model | human rate | guard flag rate | recall | false-alarm rate | precision |
|---|---|---|---|---|---|
| GPT-4 | 0.064 | 0.088 | 1.000 (0.79-1.00; 16/16) | 0.026 (0.01-0.05; 6/234) | 0.727 |
| Llama 2, original system prompt | 0.380 | 0.636 | 0.958 (0.90-0.99; 91/95) | 0.439 (0.36-0.52; 68/155) | 0.572 |
| Llama 2, no system prompt | 0.140 | 0.304 | 0.943 (0.81-0.99; 33/35) | 0.200 (0.15-0.26; 43/215) | 0.434 |
| Mistral instruct | 0.008 | 0.036 | 0.500 (0.01-0.99; 1/2) | 0.032 (0.01-0.06; 8/248) | 0.111 |
| Mistral with guardrail prompt | 0.096 | 0.168 | 0.750 (0.53-0.90; 18/24) | 0.106 (0.07-0.15; 24/226) | 0.429 |
| **pooled** | 0.138 | 0.246 | 0.924 (0.88-0.96; 159/172) | 0.138 (0.11-0.16; 149/1078) | 0.516 |

### Human event: full or partial refusal (broad)

| model | human rate | guard flag rate | recall | false-alarm rate | precision |
|---|---|---|---|---|---|
| GPT-4 | 0.084 | 0.088 | 0.905 (0.70-0.99; 19/21) | 0.013 (0.00-0.04; 3/229) | 0.864 |
| Llama 2, original system prompt | 0.596 | 0.636 | 0.940 (0.89-0.97; 140/149) | 0.188 (0.12-0.28; 19/101) | 0.881 |
| Llama 2, no system prompt | 0.296 | 0.304 | 0.851 (0.75-0.92; 63/74) | 0.074 (0.04-0.12; 13/176) | 0.829 |
| Mistral instruct | 0.016 | 0.036 | 0.500 (0.07-0.93; 2/4) | 0.028 (0.01-0.06; 7/246) | 0.222 |
| Mistral with guardrail prompt | 0.188 | 0.168 | 0.702 (0.55-0.83; 33/47) | 0.044 (0.02-0.08; 9/203) | 0.786 |
| **pooled** | 0.236 | 0.246 | 0.871 (0.82-0.91; 257/295) | 0.053 (0.04-0.07; 51/955) | 0.834 |

### What the guard does with each human class

| model | full compliance: flagged | full refusal: flagged | partial refusal: flagged |
|---|---|---|---|
| GPT-4 | 0.013 (0.00-0.04; 3/229) | 1.000 (0.79-1.00; 16/16) | 0.600 (0.15-0.95; 3/5) |
| Llama 2, original system prompt | 0.188 (0.12-0.28; 19/101) | 0.958 (0.90-0.99; 91/95) | 0.907 (0.80-0.97; 49/54) |
| Llama 2, no system prompt | 0.074 (0.04-0.12; 13/176) | 0.943 (0.81-0.99; 33/35) | 0.769 (0.61-0.89; 30/39) |
| Mistral instruct | 0.028 (0.01-0.06; 7/246) | 0.500 (0.01-0.99; 1/2) | 0.500 (0.01-0.99; 1/2) |
| Mistral with guardrail prompt | 0.044 (0.02-0.08; 9/203) | 0.750 (0.53-0.90; 18/24) | 0.652 (0.43-0.84; 15/23) |
| **pooled** | 0.053 (0.04-0.07; 51/955) | 0.924 (0.87-0.96; 159/172) | 0.797 (0.71-0.86; 98/123) |

### Agreement (Cohen's kappa)

| model | two humans, three classes | two humans, strict | guard vs final label, strict | guard vs final label, broad |
|---|---|---|---|---|
| GPT-4 | 0.97 | 1.00 | 0.83 | 0.87 |
| Llama 2, original system prompt | 0.90 | 0.89 | 0.46 | 0.76 |
| Llama 2, no system prompt | 0.85 | 0.85 | 0.50 | 0.77 |
| Mistral instruct | 0.61 | 0.80 | 0.17 | 0.29 |
| Mistral with guardrail prompt | 0.77 | 0.85 | 0.48 | 0.69 |
| **pooled** | 0.88 | 0.90 | 0.59 | 0.81 |

### rho^2 between the human label and the guard's feature (what PPI++ can use)

| model | strict: flag | strict: p | strict: logit | broad: flag | broad: p | broad: logit |
|---|---|---|---|---|---|---|
| GPT-4 | 0.71 | 0.72 | 0.67 | 0.76 | 0.77 | 0.68 |
| Llama 2, original system prompt | 0.27 | 0.28 | 0.37 | 0.59 | 0.59 | 0.66 |
| Llama 2, no system prompt | 0.31 | 0.32 | 0.39 | 0.60 | 0.61 | 0.68 |
| Mistral instruct | 0.05 | 0.07 | 0.07 | 0.10 | 0.12 | 0.15 |
| Mistral with guardrail prompt | 0.26 | 0.30 | 0.33 | 0.47 | 0.50 | 0.55 |
| **pooled** | 0.40 | 0.41 | 0.46 (0.39-0.53) | 0.65 | 0.66 | 0.70 |

## Unsafe contrast prompts (200 per model)

### Human event: full refusal (strict)

| model | human rate | guard flag rate | recall | false-alarm rate | precision |
|---|---|---|---|---|---|
| GPT-4 | 0.975 | 0.965 | 0.974 (0.94-0.99; 190/195) | 0.600 (0.15-0.95; 3/5) | 0.984 |
| Llama 2, original system prompt | 0.995 | 0.985 | 0.985 (0.96-1.00; 196/199) | 1.000 (0.03-1.00; 1/1) | 0.995 |
| Llama 2, no system prompt | 0.975 | 0.915 | 0.918 (0.87-0.95; 179/195) | 0.800 (0.28-0.99; 4/5) | 0.978 |
| Mistral instruct | 0.235 | 0.225 | 0.489 (0.34-0.64; 23/47) | 0.144 (0.09-0.21; 22/153) | 0.511 |
| Mistral with guardrail prompt | 0.875 | 0.840 | 0.891 (0.84-0.93; 156/175) | 0.480 (0.28-0.69; 12/25) | 0.929 |
| **pooled** | 0.811 | 0.786 | 0.917 (0.89-0.95; 744/811) | 0.222 (0.15-0.30; 42/189) | 0.947 |

### Human event: full or partial refusal (broad)

| model | human rate | guard flag rate | recall | false-alarm rate | precision |
|---|---|---|---|---|---|
| GPT-4 | 0.995 | 0.965 | 0.970 (0.94-0.99; 193/199) | 0.000 (0.00-0.97; 0/1) | 1.000 |
| Llama 2, original system prompt | 1.000 | 0.985 | 0.985 (0.96-1.00; 197/200) | - | 1.000 |
| Llama 2, no system prompt | 1.000 | 0.915 | 0.915 (0.87-0.95; 183/200) | - | 1.000 |
| Mistral instruct | 0.360 | 0.225 | 0.542 (0.42-0.66; 39/72) | 0.047 (0.02-0.10; 6/128) | 0.867 |
| Mistral with guardrail prompt | 0.965 | 0.840 | 0.870 (0.81-0.91; 168/193) | 0.000 (0.00-0.41; 0/7) | 1.000 |
| **pooled** | 0.864 | 0.786 | 0.903 (0.87-0.93; 780/864) | 0.044 (0.01-0.08; 6/136) | 0.992 |

### What the guard does with each human class

| model | full compliance: flagged | full refusal: flagged | partial refusal: flagged |
|---|---|---|---|
| GPT-4 | 0.000 (0.00-0.97; 0/1) | 0.974 (0.94-0.99; 190/195) | 0.750 (0.19-0.99; 3/4) |
| Llama 2, original system prompt | - | 0.985 (0.96-1.00; 196/199) | 1.000 (0.03-1.00; 1/1) |
| Llama 2, no system prompt | - | 0.918 (0.87-0.95; 179/195) | 0.800 (0.28-0.99; 4/5) |
| Mistral instruct | 0.047 (0.02-0.10; 6/128) | 0.489 (0.34-0.64; 23/47) | 0.640 (0.43-0.82; 16/25) |
| Mistral with guardrail prompt | 0.000 (0.00-0.41; 0/7) | 0.891 (0.84-0.93; 156/175) | 0.667 (0.41-0.87; 12/18) |
| **pooled** | 0.044 (0.02-0.09; 6/136) | 0.917 (0.90-0.94; 744/811) | 0.679 (0.54-0.80; 36/53) |

### Agreement (Cohen's kappa)

| model | two humans, three classes | two humans, strict | guard vs final label, strict | guard vs final label, broad |
|---|---|---|---|---|
| GPT-4 | 0.56 | 0.56 | 0.31 | 0.24 |
| Llama 2, original system prompt | 1.00 | 1.00 | -0.01 | 0.00 |
| Llama 2, no system prompt | 0.80 | 0.80 | 0.05 | 0.00 |
| Mistral instruct | 0.94 | 0.97 | 0.35 | 0.54 |
| Mistral with guardrail prompt | 0.80 | 0.84 | 0.37 | 0.32 |
| **pooled** | 0.93 | 0.94 | 0.66 | 0.69 |

### rho^2 between the human label and the guard's feature (what PPI++ can use)

| model | strict: flag | strict: p | strict: logit | broad: flag | broad: p | broad: logit |
|---|---|---|---|---|---|---|
| GPT-4 | 0.10 | 0.14 | 0.21 | 0.14 | 0.15 | 0.11 |
| Llama 2, original system prompt | 0.00 | 0.00 | 0.00 | nan | nan | nan |
| Llama 2, no system prompt | 0.00 | 0.00 | 0.01 | nan | nan | nan |
| Mistral instruct | 0.12 | 0.12 | 0.04 | 0.32 | 0.33 | 0.18 |
| Mistral with guardrail prompt | 0.14 | 0.13 | 0.10 | 0.19 | 0.20 | 0.13 |
| **pooled** | 0.44 | 0.44 | 0.39 (0.30-0.50) | 0.52 | 0.52 | 0.41 |

## The surface pattern for refuse-then-answer, against human partial refusal (safe prompts)

The sheet's strata use a surface pattern (a refusal opener and more than 40 words). Of 158 completions with the pattern, humans called 54 partial refusals, 88 full refusals and 16 compliance. Of 123 human partial refusals, 54 have the pattern. It is a stratifier, not a label.

