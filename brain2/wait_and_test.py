import time
import os
import subprocess

log_file = "/Users/pranay./.gemini/antigravity-ide/brain/39e18ada-5aea-4d63-a928-9cc9855a1772/.system_generated/tasks/task-16770.log"

print("Waiting for training to complete...")
while True:
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            content = f.read()
            if "Training Complete" in content:
                print("Training Complete detected!")
                break
    time.sleep(10)

print("Running test_brain_full.py...")
subprocess.run(["../venv2/bin/python", "test_brain_full.py"], check=True)
