import re
import json

def parse_log_line(line):
    """
    解析单条日志，提取 rank、调用栈、操作类型、cost、st、ed 等信息。
    """
    # 正则表达式匹配有效日志
    log_pattern = re.compile(
        r"(?P<rank>rank\d+)-(?P<call_stack>[\w-]+)\s+"  # rank 和调用链
        r"(?P<operation>\w+)\s+"  # 操作类型
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
    将解析后的日志转换为 Chrome Tracing 支持的 JSON 格式。
    """
    tracing_events = []
    event_id_counter = 0  # 用于生成唯一的事件 ID

    for log in parsed_logs:
        rank = log["rank"]
        call_stack = log["call_stack"]
        st = int(log["st"] * 1e3)  # 转换为微秒（Chrome Tracing 要求时间单位为微秒）
        ed = int(log["ed"] * 1e3)
        operation=log["operation"]
        # name = "->".join(call_stack)  # 使用调用栈的完整路径作为事件名称
        name = call_stack[-1]

        # 生成唯一的事件 ID
        event_id = event_id_counter
        event_id_counter += 1
        tracing_events.append({
            "name": name,
            "cat": "function",  # 类别（可选）
            "ph": "X",  # 事件类型：B 表示开始
            "ts": st,  # 时间戳（微秒）
            "dur": ed-st,
            "pid": rank,  # 进程 ID
            "tid": operation,  # 线程 ID（默认为 main）
            "id": event_id,  # 唯一事件 ID
            "args": {
                "call_stack": call_stack  # 调用栈信息
            }
        })
        # # 添加开始事件 (B)
        # tracing_events.append({
        #     "name": name,
        #     "cat": "function",  # 类别（可选）
        #     "ph": "B",  # 事件类型：B 表示开始
        #     "ts": st,  # 时间戳（微秒）
        #     "pid": rank,  # 进程 ID
        #     "tid": "main",  # 线程 ID（默认为 main）
        #     "id": event_id,  # 唯一事件 ID
        #     "args": {
        #         # "call_stack": call_stack  # 调用栈信息
        #     }
        # })

        # # 添加结束事件 (E)
        # tracing_events.append({
        #     "name": name,
        #     "cat": "function",
        #     "ph": "E",  # 事件类型：E 表示结束
        #     "ts": ed,
        #     "pid": rank,
        #     "tid": "main",
        #     "id": event_id,  # 唯一事件 ID
        #     "args": {}
        # })

    return tracing_events

def process_log_file(log_path, output_json_path):
    """
    处理日志文件，提取有效日志并保存为 Chrome Tracing 格式的 JSON 文件。
    """
    parsed_logs = []

    # 读取日志文件
    with open(log_path, "r") as file:
        for line in file:
            line = line.strip()  # 去除首尾空白字符
            if not line:
                continue  # 跳过空行

            # 检查是否包含有效日志的关键字（cost st ed）
            if "cost" in line and "st" in line and "ed" in line:
                parsed_log = parse_log_line(line)
                if parsed_log:
                    parsed_logs.append(parsed_log)

    # 转换为 Chrome Tracing 格式
    tracing_events = convert_to_tracing_format(parsed_logs)

    # 将结果保存为 JSON 文件
    with open(output_json_path, "w") as json_file:
        json.dump(tracing_events, json_file, indent=4)

    print(f"Processed {len(parsed_logs)} logs. Saved to {output_json_path}.")

if __name__ == "__main__":
    # 示例调用
    log_path = "./tmp/log.log"  # 日志文件路径
    output_json_path = "./tmp/tracing_logs.json"  # 输出 JSON 文件路径
    process_log_file(log_path, output_json_path)