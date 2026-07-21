#!/usr/bin/env python3
"""
word_math.py — controlled arithmetic word problems (language -> operation).

The brain already computes (the search engine solves algebra). The hard part was
never the math — it's mapping a sentence to the operation. This is a grammar
extractor: pull the numbers, detect the operator from a verb (give/lose ->
subtract, get/buy -> add), then compute and show the arithmetic.

    solve("I have 10 apples and give 3 away, how many do I have left?")
        -> "7 apples. (10 - 3 = 7)"

Honest scope: controlled single-operation problems (add / subtract) over a
running quantity. Multi-step, rates, ratios, or free phrasing need a real
semantic parser (an LLM). Returns None when it can't parse one cleanly.
"""

import re

NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
# verbs that REMOVE from the running quantity vs ADD to it
SUBTRACT = {"give", "gave", "given", "lose", "lost", "spend", "spent", "eat",
            "ate", "sell", "sold", "remove", "removed", "drop", "dropped",
            "away", "take", "took", "use", "used", "break", "broke", "minus"}
ADD = {"get", "got", "buy", "bought", "add", "added", "receive", "received",
       "gain", "gained", "find", "found", "more", "pick", "picked", "earn",
       "earned", "collect", "collected", "plus"}


def _tokens(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def _numbers(tokens):
    nums = []
    for t in tokens:
        if t.isdigit():
            nums.append(int(t))
        elif t in NUM_WORDS:
            nums.append(NUM_WORDS[t])
    return nums


def _item(text):
    """The noun right after the first quantity (the thing being counted)."""
    m = re.search(r"(\d+|" + "|".join(NUM_WORDS) + r")\s+([a-z]+)", text.lower())
    return m.group(2) if m else None


def solve(text):
    """Return a worked answer string, or None if it isn't a clean +/- problem."""
    tokens = _tokens(text)
    nums = _numbers(tokens)
    if len(nums) < 2:
        return None
    toks = set(tokens)
    if toks & SUBTRACT:
        result = nums[0] - sum(nums[1:])
        expr = f"{nums[0]} - " + " - ".join(str(n) for n in nums[1:])
    elif toks & ADD:
        result = sum(nums)
        expr = " + ".join(str(n) for n in nums)
    else:
        return None
    item = _item(text)
    noun = f" {item}" if item else ""
    # keep the noun plural-ish as given; just report the count and the working
    return f"{result}{noun}. ({expr} = {result})"


def _demo():
    print("word_math demo:")
    for q in ["I have 10 apples and give 3 away, how many do I have left?",
              "I have 5 marbles and find 4 more, how many do I have?",
              "There are twelve cookies and I eat five, how many are left?",
              "She has 3 cats."]:
        print(f"  > {q}")
        print(f"    {solve(q)}")


if __name__ == "__main__":
    _demo()
