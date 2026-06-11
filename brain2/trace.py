import sys

def trace_calls(frame, event, arg):
    if event == "call" and frame.f_code.co_filename.endswith("train_rl.py"):
        print(f"Call to {frame.f_code.co_name} at {frame.f_lineno} of {frame.f_code.co_filename}")
    return trace_calls

import train_rl
sys.settrace(trace_calls)
train_rl.train_rl()
