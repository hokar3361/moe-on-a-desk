# Reproducer for consecutive requests that share a prompt prefix (GitHub issue #1 of moe-on-a-desk):
# a long fixed system prompt, then different user turns, sampled (temperature > 0) and greedy, one request at a time.
# usage: mtrepro.py [port=8080] [rounds=6]
import json, sys, urllib.request

port = sys.argv[1] if len(sys.argv) > 1 else "8080"
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 6

system = ("You are a companion in a block-building adventure game. You look at the scenery and the items the player "
          "carries and write short poems about them. Always answer in JSON with the keys \"lines\" and \"reason\". ") * 12
scenes = ["a grassy field with an oak fence and three eggs under a blue sky",
          "a dark mine shaft lit by a torch, coal ore and bones on the floor",
          "a snowy slope at dusk, a fox and a lantern near a spruce tree",
          "a desert well at noon with cactus and a sleeping camel",
          "a river bank at night with lily pads and a small boat"]

def post(body):
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=600).read().decode("utf-8"))

for i in range(rounds):
    scene = scenes[i % len(scenes)]
    for temp, seed in ((0.6, 7 + i), (0.0, 1000 + i)):
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": f"Scene: {scene}. Write a three-line poem in Japanese (5-7-5 morae)."}]
        if temp == 0.0:   # a follow-up turn that extends the previous conversation
            msgs += [{"role": "assistant", "content": '{"lines": ["..."], "reason": "draft"}'},
                     {"role": "user", "content": "Check the poem against the scene and rewrite the weakest line."}]
        res = post({"model": "m", "messages": msgs, "temperature": temp, "seed": seed, "max_tokens": 300,
                    "chat_template_kwargs": {"enable_thinking": False}})
        t = res.get("timings", {})
        print(f"round {i} temp {temp}: {t.get('predicted_n')} tokens, {t.get('predicted_per_second', 0):.1f} tok/s, "
              f"cached {t.get('cache_n')}, prompt {t.get('prompt_n')}", flush=True)
