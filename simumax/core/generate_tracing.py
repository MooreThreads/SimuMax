import re
import json

def parse_log_line(line):
    """
    Parse a single log line and extract rank, call stack, operation type, cost, st, ed, etc.
    """
    # Regex matching a valid log line
    log_pattern = re.compile(
        r"(?P<rank>rank\d+)-(?P<call_stack>[\w-]+)\s+"  # rank and call chain
        r"(?P<operation>\w+)\s+"  # operation type
        r"cost\s+(?P<cost>\d+\.\d+)\s+"  # cost
        r"st\s+(?P<st>\d+\.\d+)\s+"  # st
        r"ed\s+(?P<ed>\d+\.\d+)"  # ed
    )
    match = log_pattern.match(line)
    if match:
        rank = match.group("rank")
        call_stack = match.group("call_stack").split("-")
        operation = match.group("operation")
        cost = float(match.group("cost"))
        st = float(match.group("st"))
        ed = float(match.group("ed"))

        return {
            "rank": rank,
            "call_stack": call_stack,
            "operation": operation,
            "cost": cost,
            "st": st,
            "ed": ed
        }
    return None

def convert_to_tracing_format(parsed_logs):
    """
    Convert parsed logs to the JSON format expected by Chrome Tracing.
    """
    tracing_events = []
    event_id_counter = 0  # Used to generate unique event IDs

    for log in parsed_logs:
        rank = log["rank"]
        call_stack = log["call_stack"]
        st = int(log["st"] * 1e3)  # Convert to microseconds (Chrome Tracing uses microseconds)
        ed = int(log["ed"] * 1e3)
        operation=log["operation"]
        # name = "->".join(call_stack)  # Use the full call stack path as the event name
        name = call_stack[-1]

        # Generate a unique event ID
        event_id = event_id_counter
        event_id_counter += 1
        tracing_events.append({
            "name": name,
            "cat": "function",  # Category (optional)
            "ph": "X",  # Event phase: B denotes begin
            "ts": st,  # Timestamp (microseconds)
            "dur": ed-st,
            "pid": rank,  # Process ID
            "tid": operation,  # Thread ID (defaults to main)
            "id": event_id,  # Unique event ID
            "args": {
                "call_stack": call_stack  # Call stack info
            }
        })
        # # Add begin event (B)
        # tracing_events.append({
        #     "name": name,
        #     "cat": "function",  # Category (optional)
        #     "ph": "B",  # Event phase: B denotes begin
        #     "ts": st,  # Timestamp (microseconds)
        #     "pid": rank,  # Process ID
        #     "tid": "main",  # Thread ID (defaults to main)
        #     "id": event_id,  # Unique event ID
        #     "args": {
        #         # "call_stack": call_stack  # Call stack info
        #     }
        # })

        # # Add end event (E)
        # tracing_events.append({
        #     "name": name,
        #     "cat": "function",
        #     "ph": "E",  # Event phase: E denotes end
        #     "ts": ed,
        #     "pid": rank,
        #     "tid": "main",
        #     "id": event_id,  # Unique event ID
        #     "args": {}
        # })

    return tracing_events

def process_log_file(log_path, output_json_path):
    """
    Process a log file, extract valid logs, and save them as a Chrome Tracing JSON file.
    """
    parsed_logs = []

    # Read the log file
    with open(log_path, "r") as file:
        for line in file:
            line = line.strip()  # Strip leading/trailing whitespace
            if not line:
                continue  # Skip empty lines

            # Check whether the line contains the required keywords (cost st ed)
            if "cost" in line and "st" in line and "ed" in line:
                parsed_log = parse_log_line(line)
                if parsed_log:
                    parsed_logs.append(parsed_log)

    # Convert to Chrome Tracing format
    tracing_events = convert_to_tracing_format(parsed_logs)

    # Save the result as a JSON file
    with open(output_json_path, "w") as json_file:
        json.dump(tracing_events, json_file, indent=4)

    print(f"Processed {len(parsed_logs)} logs. Saved to {output_json_path}.")

if __name__ == "__main__":
    # Example usage
    log_path = "./tmp/log.log"  # Log file path
    output_json_path = "./tmp/tracing_logs.json"  # Output JSON file path
    process_log_file(log_path, output_json_path)
