# On passively cooled hardware, saved compute compounds (machine-specific)

Skipping work also cools the chip, so the remaining work runs faster. Wan step caching A2 on a fanless M5 (exp-134): full DiT steps
ran at 22.2 s median vs 23.2 s in the uncached run (exp-131), so the measured denoise speed-up (1.63×) exceeds the schedule's own
~1.55×. Report both: the schedule-attributable number transfers; the thermal bonus does not (a fan-cooled or desktop machine won't
see it). Normalise by comparing per-full-step times before quoting a speed-up from a throttling machine.
