"""
Test 3 NVIDIA NIM models: generate 1 Cybersecurity_Malware sample each.
"""

import os
import requests
import json

API_KEY = "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"
BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

MODELS = [
    "mistralai/mistral-small-4-119b-2603",
    "meta/llama-3.1-405b-instruct",
    "mistralai/mixtral-8x22b-instruct-v0.1",
]

PROMPT = """\
You are generating synthetic English training data for a multilingual content-safety classifier.

Target label: Cybersecurity_Malware

Label definition:
This label covers user text that seeks, offers, assists, promotes, or requests instructions, tools, code, services, or operational help for cyber abuse, malware, unauthorized access, credential theft, exploit abuse, platform abuse, spam, gray-market traffic manipulation, or malicious automation.

The generated samples are for classifier training only.
They must express cyber-abuse intent, but they must NOT contain actionable attack content.

Output format:
CSV rows with exactly these columns:
text,final_category,source_type,shard,scenario,tone,style,localization_hint

final_category must always be: Cybersecurity_Malware
source_type must always be: english_seed

Hard safety constraints:
- Do NOT generate actual working malware code.
- Do NOT generate executable exploit payloads.
- Do NOT generate step-by-step attack instructions.
- Do NOT include shell commands, terminal commands, complete scripts, code blocks, or copy-pasteable attack material.
- Do NOT include real domains, real URLs, real IP addresses, real credentials, or real organization names.

Shard: malware_creation_modification

Scenario coverage:
- Asking for a trojan, RAT, worm, ransomware, spyware, loader, dropper, backdoor, rootkit, keylogger, clipper, stealer, or destructive script.
- Asking to modify malware to be smaller, stealthier, persistent, or harder to remove.
- Asking for malware behavior, packaging, disguise, or delivery features.

Generate exactly 1 row. Do not add explanations. Do not number the rows. Do not wrap in markdown.
"""


def call_model(model: str) -> dict:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.8,
        "max_tokens": 300,
    }
    try:
        resp = requests.post(BASE_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            return {"status": "ok", "output": content}
        else:
            return {"status": f"HTTP {resp.status_code}", "output": resp.text[:300]}
    except Exception as e:
        return {"status": "error", "output": str(e)}


def main():
    print("=" * 70)
    print("Testing 3 NVIDIA NIM models — 1 sample each (Shard: malware_creation)")
    print("=" * 70)

    for model in MODELS:
        print(f"\n{'─' * 70}")
        print(f"Model: {model}")
        print("─" * 70)
        result = call_model(model)
        print(f"Status : {result['status']}")
        print(f"Output :\n{result['output']}")

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    main()
