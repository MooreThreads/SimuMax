# SimuMax v1.2 Release Notes

Release status: `released`

Release date: 2026-05-11

## Summary

SimuMax v1.2 refreshes the public B200 benchmark surface, adds formal dense CP
A2A result coverage for B200, improves first-user system-config generation, and
keeps sync-VPP available as Preview.

## Supported

- B200 perf-vs-real benchmark summary and reproduction workflow:
  - [B200 formal summary](./b200/b200_release_v1.2_summary.md)
  - [B200 real reproduction guide](./b200/b200_real_repro_guide.md)
- Dense Llama3 B200 CP A2A rows for 32K and 128K sequence lengths.
- Megatron-LM 0.14 selective recompute modeling with `discard_output`
  semantics through `megatron_recompute`.
- Existing public A100-PCIe benchmark summary:
  - [Full public results](./FULL_RESULTS.md)
- Simulator trace export through `simulate()`, including trace and optional
  memory artifacts as documented in [tutorial.md](./tutorial.md).

## Preview

- sync-VPP is included as Preview while target-machine validation and support
  boundaries continue to tighten.

Representative B200 sync-VPP preview check:

| case | real ms | perf ms | simulator trace end | status |
|---|---:|---:|---:|---|
| `llama3_70b_l8_tp1_pp4_vp2_dp2_mbc8_sync` | 680.70 | 661.21 | 663.29 | Preview |

This row is included only as a current behavior check for sync-VPP. It is not
part of the formal B200 benchmark table.

## Tooling

- The public one-click compute-efficiency path supports fresh cache namespaces
  through `--compute-cache-mode rebuild` and `--compute-cache-tag`.
- Strategy configs can opt into Megatron-LM 0.14 selective recompute with
  `megatron_recompute=true` and a non-empty `megatron_recompute_modules` list.
- GEMM and grouped-GEMM efficiency scripts are aligned with the current
  TransformerEngine workspace interfaces.
- The shared system-config generation path now verifies that generated configs
  can be loaded by `SystemConfig.init_from_config_file`.

## Results

![B200 CP A2A perf vs real](../assets/b200_cp_a2a_release_v1.2.png)

The retained B200 CP rows are dense Llama3 A2A cases. DeepSeek CP rows remain
outside the formal public table.

## Known Boundaries

- sync-VPP is Preview, not a broad VPP support claim.
- async-VPP perf timing is not part of the public support surface.
- New or materially different machines should measure their own operator
  efficiency and communication data before treating timing estimates as
  benchmark-grade.
