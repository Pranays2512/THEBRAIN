import subprocess
import time

print("Starting MainRepl to test Native Biological Chat translation...\n")

p = subprocess.Popen(
    ["java", "-Djava.library.path=build_cmake", "-cp", "out_java", "brain3.MainRepl"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

def read_until_prompt():
    out = ""
    while True:
        line = p.stdout.readline()
        if not line:
            break
        out += line
        if "You: " in line:
            break
    return out

startup_text = read_until_prompt()
print(startup_text, end="")

prompt = "Hello my friend! I am having a great day today."
print(prompt)
p.stdin.write(prompt + "\n")
p.stdin.flush()

response = read_until_prompt()
print(response, end="")

p.stdin.write("quit\n")
p.stdin.flush()
p.wait()
print("\nConversation ended.")
