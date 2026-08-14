import subprocess
import time

prompts = [
    "Hey there! I'm just stopping by to chat. How's your day going?",
    "My day has been pretty good too, just working on some code and listening to music. Do you like music?",
    "That's cool! I really enjoy jazz and lo-fi beats, they help me focus.",
    "Yeah, focus is so important when coding. Anyway, what's on your mind today?",
    "That's really interesting. Well, I have to get back to work now, but it was great chatting with you!"
]

print("Starting conversation with Brain3 (Java JNI + Ollama)...\n")

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

# Wait for startup and initial prompt
startup_text = read_until_prompt()
print(startup_text, end="")

for prompt in prompts:
    print(f"{prompt}")
    p.stdin.write(prompt + "\n")
    p.stdin.flush()
    
    response = read_until_prompt()
    print(response, end="")

p.stdin.write("quit\n")
p.stdin.flush()
p.wait()
print("\nConversation ended.")
