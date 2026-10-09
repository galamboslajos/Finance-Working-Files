# Starting research workflow

Status: SETUP. New strategy research begins here; the previous attempt is a reference.

1. Audit the downloaded U.S. inputs: dated S&P 1500 membership, stable identity
   joins, accepted-price rules, availability, delisted names and corporate actions.
   Preserve missing-data denominators and the prior attempt's failure evidence.
2. Review the intended Deep Momentum paper and the prior implementation. Specify
   which elements are reproduced and which are adapted to U.S. equities.
3. Freeze the universe, sample period, feature definitions, labels, signal/entry
   timing, chronological folds, transaction costs and acceptance criteria.
4. Implement a bounded real-data feature prototype. Compare the prior five-feature
   setup with any proposed feature expansion before deciding the model inputs.
5. Specify the neural architecture and loss, with momentum, volatility-adjusted
   momentum and ridge benchmarks under the same eligible cohort and timing.
6. Measure a small engineering sample, then prepare a bounded full experiment
   with resource estimates, cost limits, output provenance and cleanup controls.

The prior project used 1998 warmup, a 1999-2023 accepted signal panel, 2009-2023
primary inputs and 2015-2023 test folds. These are reference dates, not a silently
adopted new evaluation design. Its 2024+ holdout remains protected until the new
protocol explicitly decides how it will be used.

The previous comparison's economics were inconclusive due to missing valuations
and event/identity limitations. Do not interpret accepted model fits or feature
construction as evidence of complete economic performance.
