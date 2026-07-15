# Tier-2 row-inclusion table (FROZEN before any fit; ADDENDUM 8 step 0)
Point rows (margin = geometric-mid of onset bracket, err = half bracket width):
  SGDM:  b8_s0 0.259, b32_s0 0.149, b32_s1 0.099, b64_s0 0.025, b64_s1 0.099,
         b128_s0 0.149, b128_s1 0.149, b512_s0/s1 0.025, b2048_s0/s1 0.025, b99_s0 0.643
  NEST:  b128_s0/s1 0.049, b256_s0/s1 0.025, b512_s0 0.025, b2048_s0/s1 0.025
  ADAM:  adam05_s0 0.00 [0, 0.05] (rule-breakdown: plateau itself catapults to 103 -- at-wall
         at operating; corroborated by kappa~=5.5 vs ideal edge 6)
INTERVAL-censored (excluded from point fit; sensitivity check at both ends; finer brackets queued):
  adam_b2048_s0/s1: margin in [0.05, 0.22] (own plateau catapults to 0.09; 1.15 borderline)
RIGHT-censored (enter as lower bounds; hotter brackets queued; never dropped):
  adam_b128_s0/s1 >= 0.3; nest_b8_s0/s1 >= 0.3; b8_s1 >= 0.3; b99_s1 >= 0.5
Budget rows: only where death observed (b8_s0 1.03, b32_s0 1.39, b2048 1.12x2, b512 1.12x2,
  nest_b128 1.19x2, nest_b256 1.12x2, nest_b512 1.12, nest_b2048 1.12x2); lower bounds elsewhere.
