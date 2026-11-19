# kopler-gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Kopler12/kopler-gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Kopler12/kopler-gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Trophy Tide: a forecaster for Grand Theft Auto VI trophy earn rates on Exophase. Given the house's recording of every trophy's rarity up to a checkpoint, it forecasts each trophy's earn rate at the target (a week after launch) as five quantiles (5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale.

The model works per trophy in logit space and blends three views of the same history: persistence (the last read), a recency-weighted logit trend against log-time since launch (damped for the horizon and capped), and an earners-over-players decomposition that fits the implied earner count and the tracked-player count separately and recombines them — because the rate is a ratio, and when the player base grows faster than a trophy's earners, the rate falls even while more people earn it. The spread grows with the square root of the horizon plus the trend fit's residual scale, with a wide spread when the whole history is frozen (launch night's page before its first recompute), since the first recompute can move every rate at once.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads every `<id>.in` case in the input directory and writes `<id>.out` with `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`. Deterministic: two runs on the same case write identical bytes.

---

<sub>Entered on [make.fun](https://make.fun/?ref=3t08f211&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
