"""
grade_tests_700.py — Grades results from test_results_700.txt
"""
import math
import re

with open("test_results_700.txt", "r") as f:
    lines = [line.strip() for line in f if line.strip()]

stats       = {}
failed_log  = []

for line in lines:
    if not line.startswith("["): continue

    bracket_end = line.index("]")
    feature     = line[1:bracket_end]
    rest        = line[bracket_end+2:].split(" | A: ")
    if len(rest) < 2: continue
    q = rest[0].replace("Q: ", "").strip()
    a = rest[1].strip()

    if feature not in stats:
        stats[feature] = {"passed": 0, "failed": 0, "total": 0}

    stats[feature]["total"] += 1
    is_pass = False

    # ── Universal failure signals ─────────────────────────────────────────────
    fail_signals = ["math parse error", "couldn't solve", "i don't know", "i don't know"]
    is_failure = any(s in a.lower() for s in fail_signals) or a.strip() == ""
    
    if is_failure:
        stats[feature]["failed"] += 1
        failed_log.append((feature, q, a, "returned failure / empty"))
        continue

    # ── Feature-specific correctness ─────────────────────────────────────────
    try:
        tokens = q.split()

        if feature == "algebra":
            # "a x + b = c" → x = (c-b)//a
            a_val  = int(tokens[0])
            b_val  = int(tokens[3])
            c_val  = int(tokens[5])
            true_x = (c_val - b_val) // a_val
            m = re.search(r"x\s*=\s*(-?\d+)", a)
            if m and int(m.group(1)) == true_x:
                is_pass = True

        elif feature == "permute":
            # "n permute r" → n! / (n-r)!
            n, r = int(tokens[0]), int(tokens[2])
            true_ans = math.factorial(n) // math.factorial(n - r)
            if a.strip() == str(true_ans):
                is_pass = True

        elif feature == "probability":
            # "probability of target in total" → target / total as float (2dp)
            target = int(tokens[2])
            total  = int(tokens[4])
            true_ans = f"{target / total:.2f}"
            if a.strip() == true_ans:
                is_pass = True

        elif feature == "area":
            # "area of l and w" → l * w
            l, w = int(tokens[2]), int(tokens[4])
            true_ans = l * w
            if a.strip() == str(true_ans):
                is_pass = True

        elif feature == "power":
            # "base power exp" → base ** exp
            base, exp = int(tokens[0]), int(tokens[2])
            true_ans = base ** exp
            if a.strip() == str(true_ans):
                is_pass = True

        elif feature == "semantic_query":
            # Non-empty, non-failure string = pass
            if len(a.strip()) > 0:
                is_pass = True

        elif feature == "describe":
            # At least one valid sentence returned = pass
            if "." in a and len(a) > 5:
                is_pass = True

    except Exception:
        is_pass = False

    if is_pass:
        stats[feature]["passed"] += 1
    else:
        stats[feature]["failed"] += 1
        failed_log.append((feature, q, a, "wrong answer"))

# ── Print Report ──────────────────────────────────────────────────────────────
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

print(f"\n{BOLD}{'='*55}{RESET}")
print(f"{BOLD}  BRAIN2 COMPREHENSIVE TEST REPORT{RESET}")
print(f"{BOLD}{'='*55}{RESET}")

total_passed = 0
total_cases  = 0

for feature, data in stats.items():
    p, t = data["passed"], data["total"]
    total_passed += p
    total_cases  += t
    rate = (p / t) * 100 if t else 0
    bar  = "█" * int(rate // 5) + "░" * (20 - int(rate // 5))
    color = GREEN if rate >= 80 else (YELLOW if rate >= 50 else RED)
    print(f"  {feature.upper():<20} {bar} {color}{p}/{t} ({rate:.0f}%){RESET}")

print(f"{BOLD}{'─'*55}{RESET}")
overall = (total_passed / total_cases) * 100 if total_cases else 0
color   = GREEN if overall >= 80 else (YELLOW if overall >= 50 else RED)
print(f"  {'OVERALL':<20}{'':>5} {color}{BOLD}{total_passed}/{total_cases} ({overall:.1f}%){RESET}")
print(f"{BOLD}{'='*55}{RESET}\n")

# ── Failed Samples ────────────────────────────────────────────────────────────
if failed_log:
    print(f"Sample failures (up to 15):\n")
    for feat, q, a, reason in failed_log[:15]:
        print(f"  [{feat}] Q: {q}")
        print(f"           A: {a}")
        print(f"           ↳ {reason}\n")
