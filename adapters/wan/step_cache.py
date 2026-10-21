"""Combined step + CFG reuse for the Wan DiT (one mechanism, one quality gate; Tier A, numerics-changing by design).

Per denoising step the schedule picks one mode:
  full  normal CFG pass (B=2). Records the block stack's residual R = out(last block) - in(block 0) per batch element, and the
        CFG gap D = pred_cond - pred_uncond.
  skip  all 30 blocks bypassed: block-stack output = fresh block-0 input + R from the last full step. Patch/time embeddings and the
        head still run at the current timestep. Whole-step granularity keeps the weight streamer's block order intact.
  cond  cond half only (B=1); pred_uncond = pred_cond - D from the last full step.
  auto  dynamic (A3-d): skip if the last measured change rate of R (relative L2 change between consecutive full steps, divided by
        the steps between them) is below `threshold` and the previous step was not a skip; otherwise full.
Every full step logs that change rate (self.log), so thresholds can be calibrated from any run.
Installed on top of whatever already wraps the blocks (resident per-block compile or the weight streamer). Evidence: A1 (exp-104).
"""

import mlx.core as mx


def schedule_a2(n=50):
    """Rough A2 config from A1: full warm-up 0-9; 10-33 alternate full/skip; 34-46 repeat full/skip/cond; last 3 full."""
    modes = ["full"] * n
    for i in range(10, min(34, n - 3)):
        if (i - 10) % 2 == 1:
            modes[i] = "skip"
    for i in range(34, n - 3):
        modes[i] = ("full", "skip", "cond")[(i - 34) % 3]
    return modes


def schedule_smoke(n=6):
    """Exercises every mode once: full, full, skip, cond, full, full."""
    return (["full", "full", "skip", "cond"] + ["full"] * n)[:n]


def schedule_a3a(n=50):
    """A2 with the skip window starting at step 16 (1-based) instead of 12: full 1-15; 16-34 alternate skip/full; then as A2."""
    modes = schedule_a2(n)
    for i in range(10, 15):
        modes[i] = "full"
    return modes


def schedule_dyn(n=50):
    """A3-d: full warm-up 1-11 and last 4 steps; dynamic skip decisions in between (no cond steps)."""
    return ["full"] * 11 + ["auto"] * (n - 15) + ["full"] * 4


SCHEDULES = {"a2": schedule_a2, "a3a": schedule_a3a, "dyn": schedule_dyn, "smoke": schedule_smoke}


def _b1(v):
    if isinstance(v, mx.array):
        return v[:1] if v.ndim and v.shape[0] == 2 else v
    if isinstance(v, (list, tuple)):
        return type(v)(_b1(u) for u in v)
    return v


class StepCache:
    def __init__(self, model, modes, threshold=None):
        self.model, self.modes, self.threshold = model, modes, threshold
        self.log, self.rate, self.last_full, self.prev_mode, self.taken = [], None, None, None, []
        self.step, self.mode = 0, "full"
        self.R = self.D = self.x_in = None
        self.counts = {"full": 0, "skip": 0, "cond": 0}
        self._patched = []
        self._logged = False

    def attach(self):
        ctl = self
        blocks = list(self.model.blocks)
        last = len(blocks) - 1
        for idx, b in enumerate(blocks):
            cls = type(b)
            prev = cls.__call__

            def call(self_, x, *a, _i=idx, _prev=prev, **k):
                if ctl.mode == "skip":
                    return (x.astype(mx.float32) + ctl.R).astype(x.dtype) if _i == 0 else x
                if ctl.mode != "full":
                    return _prev(self_, x, *a, **k)
                if _i == 0:
                    ctl.x_in = x
                out = _prev(self_, x, *a, **k)
                if _i == last:
                    R = out.astype(mx.float32) - ctl.x_in.astype(mx.float32)
                    if ctl.R is not None:
                        rel = (mx.sqrt(((R - ctl.R) ** 2).sum()) / mx.sqrt((ctl.R ** 2).sum())).item()
                        gap = ctl.step - ctl.last_full
                        ctl.rate = rel / gap
                        ctl.log.append({"step": ctl.step, "gap": gap, "rel": rel, "rate": ctl.rate})
                    ctl.R, ctl.last_full = R, ctl.step
                    mx.eval(ctl.R)
                    ctl.x_in = None
                return out
            b.__class__ = type(cls.__name__, (cls,), {"__call__": call})
            self._patched.append((b, cls))

        mcls = type(self.model)
        morig = mcls.__call__

        def mcall(self_, x_list, t, context, seq_len, cross_kv_caches=None, y=None, rope_cos_sin=None):
            mode = ctl.modes[ctl.step] if ctl.step < len(ctl.modes) else "full"
            if mode == "auto":
                ok = ctl.rate is not None and ctl.threshold is not None and ctl.rate < ctl.threshold and ctl.prev_mode != "skip"
                mode = "skip" if ok else "full"
            if mode != "full" and (ctl.R is None or ctl.D is None or len(x_list) != 2):
                mode = "full"
            ctl.prev_mode = mode
            ctl.taken.append(mode)
            ctl.step += 1
            ctl.mode = mode
            ctl.counts[mode] += 1
            if mode == "cond":
                if not ctl._logged:
                    print("[cache] B=1 slicing: rope", [getattr(r, "shape", None) for r in (rope_cos_sin or ())], flush=True)
                    ctl._logged = True
                out = morig(self_, x_list[:1], t=_b1(t), context=_b1(context), seq_len=seq_len,
                            cross_kv_caches=_b1(cross_kv_caches), y=_b1(y), rope_cos_sin=_b1(rope_cos_sin))
                cond = out[0]
                return [cond, cond - ctl.D]
            out = morig(self_, x_list, t=t, context=context, seq_len=seq_len, cross_kv_caches=cross_kv_caches, y=y,
                        rope_cos_sin=rope_cos_sin)
            if mode == "full":
                ctl.D = out[0] - out[1]
                mx.eval(ctl.D)
            return out
        self.model.__class__ = type(mcls.__name__, (mcls,), {"__call__": mcall})
        self._patched.append((self.model, mcls))
