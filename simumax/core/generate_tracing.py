import json
import re
import heapq


COMM_PREFIXES = (
    "send_",
    "recv_",
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all2all",
    "broadcast",
    "scatter",
    "gather",
    "reduce",
    "async_send",
    "async_recv",
    "async_wait_send",
    "async_wait_recv",
    "sync_send",
    "sync_recv",
    "sync_wait_recv",
)


def parse_log_line(line):
    """Parse one simulator log line."""
    log_pattern = re.compile(
        r"(?P<rank>rank\d+)-(?P<call_stack>[\w-]+)\s+"
        r"(?:gid\s+(?P<gid>\S+)\s+)?"
        r"(?P<operation>\w+)\s+"
        r"cost\s+(?P<cost>\d+\.\d+)\s+"
        r"st\s+(?P<st>\d+\.\d+)\s+"
        r"ed\s+(?P<ed>\d+\.\d+)"
        r"(?:\s+post\s+(?P<post>\d+\.\d+))?"
        r"(?:\s+order\s+(?P<order>-?\d+))?"
    )
    match = log_pattern.match(line)
    if not match:
        return None

    return {
        "rank": match.group("rank"),
        "call_stack": match.group("call_stack").split("-"),
        "gid": match.group("gid"),
        "operation": match.group("operation"),
        "cost": float(match.group("cost")),
        "st": float(match.group("st")),
        "ed": float(match.group("ed")),
        "post": float(match.group("post")) if match.group("post") is not None else None,
        "order": int(match.group("order")) if match.group("order") is not None else None,
    }


def _is_comm_event(event_name):
    return event_name.startswith(COMM_PREFIXES)


def _rank_sort_key(pid):
    match = re.match(r"rank(\d+)", str(pid))
    return int(match.group(1)) if match else 0


def _ordered_tid(tid):
    if str(tid).startswith("pp_detail_"):
        return f"07_{tid}"
    order = {
        "fwd_scope": "00_fwd_scope",
        "bwd_scope": "01_bwd_scope",
        "fwd_compute": "02_fwd_compute",
        "bwd_compute": "03_bwd_compute",
        "pp_fwd": "04_pp_fwd",
        "pp_bwd": "05_pp_bwd",
        "pp_batch_scope": "06_pp_batch_scope",
        "pp_detail": "07_pp_detail",
        "comm": "08_comm",
        "wait": "09_wait",
    }
    return order.get(tid, f"99_{tid}")


def _display_tid(ordered_tid):
    parts = str(ordered_tid).split("_", 1)
    return parts[1] if len(parts) == 2 else str(ordered_tid)


def _thread_sort_index(tid):
    tid = str(tid)
    if tid.startswith("07_pp_detail_"):
        suffix = tid[len("07_pp_detail_") :]
        try:
            return 60 + int(suffix)
        except ValueError:
            return 60
    order = {
        "00_fwd_scope": 0,
        "01_bwd_scope": 1,
        "02_fwd_compute": 2,
        "03_bwd_compute": 3,
        "04_pp_fwd": 4,
        "05_pp_bwd": 5,
        "06_pp_batch_scope": 6,
        "08_comm": 80,
        "09_wait": 90,
    }
    return order.get(tid, 100)


def _comm_lane(base_name):
    if base_name.startswith(("send_", "recv_")):
        return "pp_detail"
    if base_name.startswith(("async_send_next", "async_recv_prev", "async_wait_recv_prev")):
        return "pp_fwd"
    if base_name.startswith(("async_send_prev", "async_recv_next", "async_wait_recv_next")):
        return "pp_bwd"
    return "comm"


def _flow_anchor_ts(event, prefer_end=False):
    """Anchor flow markers inside the slice to avoid boundary collisions."""
    ts = float(event["ts"])
    dur = float(event.get("dur") or 0.0)
    if dur <= 0.0:
        return ts

    # Keep the original "start vs end" intent, but move the anchor just inside
    # the slice so adjacent comm events that share a boundary do not visually
    # attach to the wrong slice in the trace viewer.
    pad = min(1e-3, dur * 0.25)
    if prefer_end:
        return ts + max(0.0, dur - pad)
    return ts + pad


def _pp_display_pad_ts(event):
    """Slightly shrink PP comm slices to avoid zero-gap viewer swallowing."""
    args = event.get("args", {}) or {}
    gid = args.get("gid")
    if not gid or "send_recv-" not in str(gid):
        return 0.0

    dur = float(event.get("dur") or 0.0)
    if dur <= 0.0:
        return 0.0

    # Keep the slice shrink smaller than the flow-anchor epsilon so flows still
    # land safely inside the visible body after the display-only trim.
    return min(5e-4, dur * 0.1)


def convert_to_tracing_format(parsed_logs):
    """
    Convert parsed logs to Chrome Tracing events.
    Stream layout:
    - fwd_compute / fwd_comm
    - bwd_compute / bwd_comm
    """
    tracing_events = []
    event_id_counter = 0

    # Only assign correlation/group id to gids that have both send and recv events.
    pairable_gid = {}
    for log in parsed_logs:
        gid = log.get("gid")
        if not gid:
            continue
        call_stack = log.get("call_stack", [])
        if not call_stack:
            continue
        base = call_stack[-1]
        st = pairable_gid.setdefault(gid, {"send": False, "recv": False})
        if base.startswith(("send_", "async_send", "sync_send")):
            st["send"] = True
        if base.startswith(("recv_", "async_recv", "sync_recv")):
            st["recv"] = True
    pairable_gid = {k for k, v in pairable_gid.items() if v["send"] and v["recv"]}

    gid_to_correlation_id = {}
    corr_counter = 0

    for log in parsed_logs:
        rank = log["rank"]
        call_stack = log["call_stack"]
        operation = log["operation"]
        # Simulator timestamps are in milliseconds; Chrome tracing uses microseconds.
        st = log["st"] * 1e3
        ed = log["ed"] * 1e3
        base_name = call_stack[-1]
        gid = log.get("gid")

        if base_name.startswith("batch_pp"):
            stream_type = "scope"
        elif base_name.startswith(("async_wait_send", "async_wait_recv")):
            stream_type = "wait"
        else:
            stream_type = "comm" if _is_comm_event(base_name) else "compute"
        lane = _comm_lane(base_name) if stream_type == "comm" else stream_type
        if stream_type == "wait":
            tid = "wait"
        elif stream_type == "comm":
            tid = lane
        elif stream_type == "scope" and base_name.startswith("batch_pp"):
            tid = "pp_batch_scope"
        else:
            tid = "bwd_compute" if operation == "recompute_fwd" else f"{operation}_{stream_type}"
        cat = stream_type
        corr_id = None
        if stream_type == "comm" and gid in pairable_gid:
            if gid not in gid_to_correlation_id:
                gid_to_correlation_id[gid] = corr_counter
                corr_counter += 1
            corr_id = gid_to_correlation_id[gid]
        name = base_name
        if corr_id is not None:
            # Visual grouping without flow lines: same comm pair gets same g<id>.
            name = f"{base_name}[g{corr_id}]"

        event_id = event_id_counter
        event_id_counter += 1
        tracing_events.append(
            {
                "name": name,
                "cat": cat,
                "ph": "X",
                "ts": st,
                "dur": max(0.0, ed - st),
                "pid": rank,
                "tid": tid,
                "id": event_id,
                "args": {
                    "call_stack": call_stack,
                    "stream_type": stream_type,
                    "lane": lane,
                    "lane_base": lane,
                    "direction": operation,
                    "gid": gid,
                    "correlation_id": corr_id,
                    "base_name": base_name,
                    "post_ts": (log.get("post") * 1e3) if log.get("post") is not None else None,
                    "post_order": log.get("order"),
                },
            }
        )

    # Overlapping sync detail send/recv events need separate visual sub-lanes,
    # otherwise flow markers can attach ambiguously when multiple events share
    # the same pid/tid and overlap in time.
    detail_by_pid = {}
    for event in tracing_events:
        if event.get("ph") != "X" or event.get("cat") != "comm":
            continue
        if (event.get("args", {}) or {}).get("lane") != "pp_detail":
            continue
        detail_by_pid.setdefault(event["pid"], []).append(event)

    for _, events in detail_by_pid.items():
        events.sort(key=lambda event: (event["ts"], event.get("dur") or 0.0, event.get("id", 0)))
        active = []  # heap[(end_ts, lane_idx)]
        free_lanes = []
        next_lane = 0
        for event in events:
            start = float(event["ts"])
            while active and active[0][0] <= start + 1e-9:
                _, lane_idx = heapq.heappop(active)
                heapq.heappush(free_lanes, lane_idx)
            if free_lanes:
                lane_idx = heapq.heappop(free_lanes)
            else:
                lane_idx = next_lane
                next_lane += 1
            event["tid"] = f"pp_detail_{lane_idx}"
            event["args"]["lane"] = event["tid"]
            event["args"]["detail_lane_idx"] = lane_idx
            heapq.heappush(active, (start + float(event.get("dur") or 0.0), lane_idx))

    # When async PP comm has a known local post time and the rank's comm stream
    # is idle, pull the displayed event start left to that post time.
    # This removes false bubbles without changing end-to-end pairing or
    # introducing overlap on the single comm lane.
    comm_by_pid_lane = {}
    for event in tracing_events:
        if event.get("cat") == "comm" and event.get("ph") == "X":
            comm_by_pid_lane.setdefault((event["pid"], event.get("tid")), []).append(event)

    for _, events_on_rank in comm_by_pid_lane.items():
        events_on_rank.sort(key=lambda event: (event["ts"], event.get("id", 0)))
        prev_end = None
        for event in events_on_rank:
            args = event.get("args", {}) or {}
            post_ts = args.get("post_ts")
            if post_ts is None:
                prev_end = event["ts"] + (event.get("dur") or 0.0)
                continue
            original_ts = float(event["ts"])
            original_end = original_ts + (event.get("dur") or 0.0)
            post_ts = float(post_ts)
            candidate_ts = post_ts if prev_end is None else max(post_ts, prev_end)
            if candidate_ts < original_ts:
                event["ts"] = candidate_ts
                event["dur"] = max(0.0, original_end - candidate_ts)
            prev_end = event["ts"] + (event.get("dur") or 0.0)

    # Display-only polish for PP p2p events: shrink the visible slice slightly
    # so tightly adjacent zero-gap comm events remain individually visible in
    # trace viewers. Keep this smaller than the flow anchor epsilon.
    for event in tracing_events:
        if event.get("cat") != "comm" or event.get("ph") != "X":
            continue
        pad = _pp_display_pad_ts(event)
        if pad <= 0.0:
            continue
        ts = float(event["ts"])
        dur = float(event.get("dur") or 0.0)
        if dur <= 2.0 * pad:
            continue
        event["ts"] = ts + pad
        event["dur"] = max(0.0, dur - 2.0 * pad)

    # Split inclusive parent/module envelopes from leaf compute events.
    # This keeps real kernels on compute lanes while preserving hierarchy on a
    # dedicated scope lane.
    compute_by_pid = {}
    for idx, event in enumerate(tracing_events):
        if event.get("cat") != "compute" or event.get("ph") != "X":
            continue
        compute_by_pid.setdefault(event["pid"], []).append((idx, event))

    for _, items in compute_by_pid.items():
        items.sort(key=lambda x: (x[1]["ts"], -(x[1]["dur"] or 0.0)))
        for i, (idx, event) in enumerate(items):
            parent_stack = event.get("args", {}).get("call_stack", [])
            if not parent_stack:
                continue
            parent_end = event["ts"] + (event.get("dur") or 0.0)
            base_name = event.get("args", {}).get("base_name", "")
            is_scope = base_name in ("recompute_block", "checkpoint_bwd")
            for j in range(i + 1, len(items)):
                child = items[j][1]
                child_stack = child.get("args", {}).get("call_stack", [])
                if len(child_stack) <= len(parent_stack):
                    continue
                if child_stack[: len(parent_stack)] != parent_stack:
                    continue
                child_end = child["ts"] + (child.get("dur") or 0.0)
                if child["ts"] >= event["ts"] and child_end <= parent_end + 1e-9:
                    is_scope = True
                    break
            if is_scope:
                direction = event.get("args", {}).get("direction", "fwd")
                event["cat"] = "scope"
                if direction == "recompute_fwd":
                    event["tid"] = "bwd_scope"
                    if base_name == "recompute_block":
                        event["name"] = "recompute_fwd"
                else:
                    event["tid"] = f"{direction}_scope"

    # For PP point-to-point comm, emit one direct flow per pair. In async mode
    # the anchor is nudged inside the slice when post_ts exists; in sync mode
    # the anchor stays on the boundary so the line only represents pair
    # identity, not extra launch semantics.
    by_gid = {}
    for event in tracing_events:
        if event.get("cat") != "comm":
            continue
        gid = event.get("args", {}).get("gid")
        if not gid or "send_recv-" not in gid:
            continue
        by_gid.setdefault(gid, []).append(event)

    flow_id = 0
    for gid, events in by_gid.items():
        sends = [
            event
            for event in events
            if event.get("args", {}).get("base_name", "").startswith(
                ("send_", "async_send", "sync_send")
            )
        ]
        recvs = [
            event
            for event in events
            if event.get("args", {}).get("base_name", "").startswith(
                ("recv_", "async_recv", "sync_recv")
            )
        ]
        if len(sends) != 1 or len(recvs) != 1:
            continue

        send = sends[0]
        recv = recvs[0]
        send_post = (send.get("args", {}) or {}).get("post_ts")
        recv_post = (recv.get("args", {}) or {}).get("post_ts")
        tracing_events.append(
            {
                "name": f"pair:{gid}",
                "cat": "comm_pair",
                "ph": "s",
                "ts": _flow_anchor_ts(send, prefer_end=False),
                "pid": send["pid"],
                "tid": send["tid"],
                "id": flow_id,
                "args": {"gid": gid},
            }
        )
        tracing_events.append(
            {
                "name": f"pair:{gid}",
                "cat": "comm_pair",
                "ph": "f",
                "ts": _flow_anchor_ts(recv, prefer_end=True),
                "pid": recv["pid"],
                "tid": recv["tid"],
                "bp": "e",
                "id": flow_id,
                "args": {"gid": gid},
            }
        )
        flow_id += 1

    for event in tracing_events:
        if "tid" in event:
            event["tid"] = _ordered_tid(event["tid"])

    process_ids = sorted({event["pid"] for event in tracing_events if "pid" in event}, key=_rank_sort_key)
    metadata_events = []
    for proc_idx, pid in enumerate(process_ids):
        metadata_events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "args": {"name": pid},
            }
        )
        metadata_events.append(
            {
                "name": "process_sort_index",
                "ph": "M",
                "pid": pid,
                "args": {"sort_index": proc_idx},
            }
        )
        metadata_events.append(
            {
                "name": "sort_index",
                "ph": "M",
                "pid": pid,
                "args": {"sort_index": proc_idx},
            }
        )
        tids = sorted({event["tid"] for event in tracing_events if event.get("pid") == pid and "tid" in event})
        for tid in tids:
            metadata_events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"name": _display_tid(tid)},
                }
            )
            metadata_events.append(
                {
                    "name": "thread_sort_index",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"sort_index": _thread_sort_index(tid)},
                }
            )
            metadata_events.append(
                {
                    "name": "sort_index",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"sort_index": _thread_sort_index(tid)},
                }
            )

    tracing_events = metadata_events + tracing_events

    return tracing_events


def process_log_file(log_path, output_json_path):
    """Read a simulator log file and write Chrome Tracing JSON."""
    parsed_logs = []

    with open(log_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            if "cost" in line and "st" in line and "ed" in line:
                parsed_log = parse_log_line(line)
                if parsed_log:
                    parsed_logs.append(parsed_log)

    tracing_events = convert_to_tracing_format(parsed_logs)

    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(tracing_events, json_file, indent=4)

    print(f"Processed {len(parsed_logs)} logs. Saved to {output_json_path}.")


if __name__ == "__main__":
    process_log_file("./tmp/log.log", "./tmp/tracing_logs.json")
