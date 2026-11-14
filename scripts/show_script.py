"""Print scripts in readable form with estimated start times.

Usage: python scripts/show_script.py [--md] FILE.json [FILE.json ...]
  --md  Markdown output: a heading per script and the timeline in a code block.
"""

import json
import sys
import textwrap


def main() -> None:
    args = sys.argv[1:]
    md = "--md" in args
    for path in [a for a in args if a != "--md"]:
        s = json.loads(open(path).read())
        if md:
            print(f"## {s['id']}\n\n```")
        print(f"=== {s['id']}  [{s['type']}]  topic {', '.join(s['topics'])}  "
              f"~{s['est_total_sec']:.0f} s total, ~{s['est_speech_sec']:.0f} s speech")
        extra = [f"wait {s['wait_sec']} s" if s.get("wait_sec") else "",
                 f"extension +{s['extension']['extra_sec']} s" if s.get("extension") else "",
                 f"friend {s['friend_name']}" if s.get("friend_name") else "",
                 f"conditions: {', '.join(s['conditions'])}" if s.get("conditions") else ""]
        print("    " + "  ".join(x for x in extra if x) + f"  voices user={s['voices']['user'][:8]}"
              + (f" friend={s['voices']['friend'][:8]}" if s['voices'].get('friend') else ""))
        ends, cur = [], 0.0
        for t in s["turns"]:
            start = ends[t["anchor"]["turn"]] + t["anchor"]["offset"] if "anchor" in t else cur + t["gap_before"]
            if start - cur >= 4:
                print(f"           ... {start - cur:.1f} s silence ...")
            ends.append(start + t["est_sec"])
            cur = max(cur, ends[-1])
            tag = t.get("event", "") + (f":{t['kind']}" if t.get("kind") and t.get("event") else "")
            tag = tag or t.get("role", "")
            if t.get("reference_only"):
                tag += " [reference only]"
            head = f"  {start:5.1f}s {t['speaker'].upper():6s}"
            body = textwrap.wrap(t["text"], 70) or [""]
            print(f"{head} {body[0]}" + (f"   <{tag}>" if tag and len(body) == 1 else ""))
            for k, line in enumerate(body[1:]):
                print(" " * len(head) + " " + line + (f"   <{tag}>" if tag and k == len(body) - 2 else ""))
        print("```\n" if md else "")


if __name__ == "__main__":
    main()
