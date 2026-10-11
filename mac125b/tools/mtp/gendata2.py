# MTP training data v2 (clean): prompts built from task x subject combinations, with every topic of the fixed metric
# and of bench-strata excluded; a separate held-out prompt set (families never used for training) for generality.
#   gen  <texts.jsonl>         : the target writes answers (greedy, thinking off)
#   dump <texts.jsonl>         : replay [user prompt, assistant answer] as an assistant continuation so the server's
#                                LLAMA_SPEC_MTP_DUMP records (token, h_mtp, h_tgt) in the original chat context
#   heldout <file.json>        : write the held-out prompt list
import json, sys, urllib.request, random, re

BANNED = re.compile(r"lighthouse|bottle|bseab|ocean|sailor|iso.?8601|bdates?b|weekday|calendar|bscores?b|brecords?b", re.I)

langs_train = ["Python", "C++", "Rust", "Go", "JavaScript", "Java", "C#", "Kotlin", "Ruby", "PHP"]
langs_held = ["Haskell", "Lua"]
code_tasks = [
    "implements an LRU cache with O(1) get and put", "parses a CSV file and prints per-column statistics",
    "merges overlapping intervals", "finds the shortest path in a weighted graph with Dijkstra's algorithm",
    "implements a thread-safe bounded queue", "converts Roman numerals to integers and back",
    "tokenizes and evaluates a simple arithmetic expression", "reads a JSON config with defaults for missing keys",
    "does binary search on a rotated sorted array", "rate-limits calls with a token bucket",
    "renders a text progress bar", "implements a trie with insert, search and prefix listing",
    "computes the edit distance between two strings", "serializes a binary tree to a string and back",
    "detects a cycle in a directed graph", "implements a min-heap priority queue",
    "solves the N-queens problem", "compresses a string with run-length encoding",
    "implements a simple key-value store with a write-ahead log", "checks whether a sudoku board is valid",
    "computes matrix multiplication for small dense matrices", "implements a bloom filter",
    "finds all anagram groups in a word list", "implements a basic HTTP request parser",
    "flattens a nested list of arbitrary depth", "generates all permutations of a list",
    "implements a circular buffer", "computes the convex hull of 2D points",
    "implements a simple regular expression matcher supporting . and *", "topologically sorts tasks with dependencies",
]
code_held_tasks = ["implements a skip list", "computes Fibonacci numbers with memoization and explains complexity",
                   "implements union-find with path compression", "validates balanced brackets in a string"]
genres_train = ["Write a short story about", "Write an essay about", "Explain, for a curious teenager,", "Write a letter about",
                "Write a dialogue between two friends about", "Write a short guide to", "Summarize what is known about",
                "Write a persuasive speech about"]
genres_held = ["Write a poem about", "Write a review of an imaginary book about"]
subjects = [
    "a clockmaker who repairs a broken town clock", "the history of the printing press", "how vaccines work",
    "a mountain village in winter", "two rival bakers preparing for a festival", "how a refrigerator works",
    "a fox and a crow who become unlikely friends", "the importance of curiosity", "how bees find flowers",
    "the first garden on Mars", "the rules of chess", "compound interest", "why the sky is blue",
    "a jazz musician's first concert", "volcanoes and how they form", "the invention of the bicycle",
    "a city library at night", "learning a second language as an adult", "the life cycle of stars",
    "a robot that learns to paint", "urban beekeeping", "the history of tea", "how batteries store energy",
    "a long train journey through the desert", "the ethics of artificial intelligence", "a forgotten family recipe",
    "how earthquakes are measured", "a street market in the morning", "the migration of birds",
    "the first day at a new job", "how the internet routes data", "a lost dog finding its way home",
    "the architecture of medieval castles", "photosynthesis", "a chess tournament between old friends",
    "the benefits of walking every day", "how glaciers shape valleys", "a small bookshop that is about to close",
    "the history of coffee", "how airplanes stay in the air", "a child's first snowfall", "the water cycle",
    "the Roman aqueducts", "a desert caravan", "how the human memory works", "an inventor's failed machine",
    "the night sky in the countryside", "the origins of mathematics", "a festival of lanterns", "how bridges are built",
]
held_subjects = ["a beekeeper in a city of glass towers", "the history of the violin", "how tides of traffic form in cities",
                 "a gardener who grows impossible flowers", "the physics of a spinning top"]

def build():
    random.seed(11)
    train = []
    for lang in langs_train:
        for t in random.sample(code_tasks, 15):
            train.append("Write a %s function that %s. Include comments and a short usage example." % (lang, t))
    for g in genres_train:
        for sub in random.sample(subjects, 25):
            train.append("%s %s." % (g, sub))
    for sub in random.sample(subjects, 30):
        train.append("Answer step by step: what are the three most important ideas behind %s, and why?" % sub)
    held = []
    for lang in langs_held:
        for t in code_held_tasks[:3]:
            held.append("Write a %s function that %s. Include comments and a short usage example." % (lang, t))
    for g in genres_held:
        for sub in held_subjects[:4]:
            held.append("%s %s." % (g, sub))
    for t in code_held_tasks:
        held.append("Write a Python function that %s. Include comments and a short usage example." % t)
    for p in train + held:
        assert not BANNED.search(p), p
    random.shuffle(train)
    return train, held[:20]

URL = "http://127.0.0.1:8080/v1/chat/completions"
def post(body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))

if __name__ == "__main__":
    train, held = build()
    mode, path = sys.argv[1], sys.argv[2]
    if mode == "heldout":
        json.dump(held, open(path, "w"), indent=1)
        print(len(train), "train prompts,", len(held), "held-out prompts")
    elif mode == "gen":
        n0 = 0
        try:
            n0 = sum(1 for _ in open(path))
        except FileNotFoundError:
            pass
        with open(path, "a") as f:
            for i, p in enumerate(train):
                if i < n0:
                    continue
                d = post({"model": "m", "messages": [{"role": "user", "content": p}], "max_tokens": 1000, "temperature": 0})
                f.write(json.dumps({"prompt": p, "text": d["choices"][0]["message"]["content"]}) + "\n"); f.flush()
                print(i, d["usage"]["completion_tokens"], "%.1f tok/s" % d["timings"]["predicted_per_second"], flush=True)
    else:
        for i, l in enumerate(open(path)):
            r = json.loads(l)
            post({"model": "m", "messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["text"]}],
                  "continue_final_message": True, "add_generation_prompt": False, "max_tokens": 1, "temperature": 0})
            if i % 20 == 0:
                print("dumped", i, flush=True)
