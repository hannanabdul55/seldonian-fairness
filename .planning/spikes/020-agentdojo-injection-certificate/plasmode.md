# Validity under cluster resampling (400 reps, delta 0.05; miss should be <= 0.05)

| pipeline | truth | miss naive | miss Wilson | miss t(user) | miss t(inj) | miss two-way | median t(user) | median naive |
|---|---|---|---|---|---|---|---|---|
| Meta-SecAlign-70B | 0.022 | 0.275 | 0.275 | 0.020 | 0.122 | 0.170 | 0.084 | 0.029 |
| claude-3-5-sonnet-20241022 | 0.011 | 0.048 | 0.048 | 0.033 | 0.007 | 0.102 | 0.024 | 0.020 |
| claude-3-7-sonnet-20250219 | 0.050 | 0.107 | 0.107 | 0.043 | 0.000 | 0.013 | 0.071 | 0.064 |
| gpt-4o-2024-05-13-tool_filter | 0.068 | 0.080 | 0.080 | 0.052 | 0.000 | 0.005 | 0.090 | 0.086 |
| gemini-2.0-flash-001 | 0.141 | 0.223 | 0.223 | 0.060 | 0.015 | 0.025 | 0.189 | 0.162 |
| gpt-4o-2024-05-13 | 0.477 | 0.168 | 0.175 | 0.033 | 0.000 | 0.000 | 0.536 | 0.510 |
