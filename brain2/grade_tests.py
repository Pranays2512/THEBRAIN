import re

with open("test_100.txt", "r") as f:
    queries = [line.strip() for line in f if line.strip()]

with open("test_results.txt", "r") as f:
    results = [line.strip() for line in f if line.startswith("You: Brain:")]

passed = 0
failed = 0
details = []

for q, r in zip(queries, results):
    r = r.replace("You: Brain: ", "").strip()
    
    is_pass = True
    reason = ""
    
    # Check for known failure strings
    if "I don't know" in r or "I couldn't solve it" in r or "Math parse error" in r or r == "":
        is_pass = False
        reason = "Returned generic failure or empty."
        
    # Check math specifically
    if "=" in q:
        # e.g. "2 x + 4 = 10" -> x = 3
        if "x =" in r:
            ans = r.split("x = ")[-1].strip()
            if not ans.isdigit():
                is_pass = False
                reason = f"Math answer was not a number: '{ans}'"
        else:
            is_pass = False
            reason = "Failed to output math result."
            
    if is_pass:
        passed += 1
    else:
        failed += 1
        details.append(f"Q: {q}\nA: {r}\nReason: {reason}\n")

print(f"Total Tests: {len(queries)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Pass Rate: {(passed/len(queries))*100:.1f}%")

if failed > 0:
    print("\n--- Failed Cases ---")
    for d in details:
        print(d)
