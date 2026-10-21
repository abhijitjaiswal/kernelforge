# Denoising change is U-shaped: the skip budget lives in the calm middle; the late rise is untouchable

Measured on Wan 2.1 1.3B (50 steps, UniPC, CFG 5; exp-a2-v3), as the relative L2 change of the DiT block stack's residual between
consecutive full steps, per step: warm-up 0.16 (step 2) → 0.024 (step 9); **calm middle 0.019–0.026 (steps 13–35)**; then a sharp
late rise 0.030 (38) → 0.051 (44) → 0.081 (47) → 0.28 (50) as fine detail is painted in.

This one curve explains the A2 failure pattern: late skips and cond-only reuse delete fine detail, so prompts whose identity is fine
detail (neon text and background figures, foam on waves) fail the LPIPS gate while smooth content (a fox in snow) passes.

Rules:
- Measure this curve first on any new model with 30+ steps (one run with the residual-change log; adapters/wan/step_cache.py logs it
  on every full step). Place reuse in the calm middle; never in the late rise.
- Do not confuse the two time axes: error along video frames (output index) is not change along denoising steps.
- A dynamic rule that skips only below a change-rate threshold encodes this by construction (rising rates stop skipping).
