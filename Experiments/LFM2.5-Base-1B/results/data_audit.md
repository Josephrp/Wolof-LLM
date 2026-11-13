# Data audit summary

## clm

- dataset: `galsenai/cleaned_data_for_clm`
- splits: `{'train': 958646, 'test': 4282}`
- length chars: `{'mean': 307.5570696874533, 'p50': 89, 'p90': 832, 'p99': 1965}`
- check: clm.train: got 958,646, expected 958,646 (±0)
- check: clm.test: got 4,282, expected 4,282 (±0)
- check: train null text rate: 0/958646
- check: test null text rate: 0/4282

## raw

- dataset: `galsenai/wolof_centalized_corpus`
- rows: 1,018,520
- check: raw.rows: got 1,018,520, expected 1,018,520 (±5000)

## sft

- dataset: `galsenai/wolof_sft`
- splits: `{'train': 990320, 'test': 2250}`
- translation %: 57.9
- tasks: `{'translation': 573832, 'tool_use': 206557, 'classification': 120701, 'instruction': 63899, 'intent': 15012, 'ner': 7505, 'qa': 2814}`
- check: sft.train: got 990,320, expected 990,320 (±9903)
- check: sft.test: got 2,250, expected 2,250 (±22)
- check: WARN translation=57.9% > 50% (soft — reweight mixture required)
- leakage: Hub test includes belebele / afrimmlu / afrimgsm / afrixnli held out of SFT train, but source sentences may still appear in the pretrain corpus. Do not claim zero leakage without an explicit CLM↔SFT overlap study.

## SFT vs CLM overlap (sample)

- `{'n_clm_test_sampled': 2000, 'n_sft_assistant_sampled': 2000, 'n_hits': 0, 'rate': 0.0, 'note': 'Exact string match on truncated assistant content vs CLM test sample.'}`

