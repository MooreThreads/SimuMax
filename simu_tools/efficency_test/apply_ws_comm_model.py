#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

OP_MAP = {
    "all_reduce": "all_reduce",
    "all_gather": "all_gather",
    "reduce_scatter": "reduce_scatter",
    "alltoall": "all2all",
}


def parse_args():
    p = argparse.ArgumentParser(description="Apply ws-specific communication model to a system json.")
    p.add_argument("--system-json", required=True)
    p.add_argument("--summary-ws2", required=True)
    p.add_argument("--summary-ws4", required=True)
    p.add_argument("--summary-ws8", required=True)
    p.add_argument("--output-json", default="")
    return p.parse_args()


def load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    args = parse_args()
    system = load(args.system_json)
    s2 = load(args.summary_ws2)["fit_compact"]
    s4 = load(args.summary_ws4)["fit_compact"]
    s8 = load(args.summary_ws8)["fit_compact"]

    for net_name, net in system.get("networks", {}).items():
        if not isinstance(net, dict) or net_name == "inter_node":
            continue
        op_dict = net.get("op", {})
        for src_op, cfg_op in OP_MAP.items():
            cfg = op_dict.get(cfg_op)
            if not isinstance(cfg, dict):
                continue
            cfg["efficient_factor"] = s8[src_op]["efficient_factor"]
            cfg["latency_us"] = s8[src_op]["latency_us"]
            cfg["efficient_factor_by_comm_num"] = {
                "2": s2[src_op]["efficient_factor"],
                "4": s4[src_op]["efficient_factor"],
                "8": s8[src_op]["efficient_factor"],
            }
            cfg["latency_us_by_comm_num"] = {
                "2": s2[src_op]["latency_us"],
                "4": s4[src_op]["latency_us"],
                "8": s8[src_op]["latency_us"],
            }

    out = Path(args.output_json) if args.output_json else Path(args.system_json)
    out.write_text(json.dumps(system, indent=4), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
