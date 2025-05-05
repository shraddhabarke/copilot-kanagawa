import subprocess
import json
import time
import re
import os

from azure.ai.inference import ChatCompletionsClient
from azure.identity import ChainedTokenCredential, DefaultAzureCredential, AzureCliCredential

"""
# === Azure CONFIG ===
scopes = ["api://trapi/.default"]
api_version = '2025-03-01-preview'
model_name = 'gpt-4o'
deployment_name = "gpt-4o_2024-11-20"
instance = "redmond/interactive/openai"
endpoint = f'https://trapi.research.microsoft.com/{instance}/deployments/' + deployment_name
"""

scopes = ["api://trapi/.default"]
api_version = '2025-03-01-preview'
model_name = 'o3'  # This should match your deployment model version
model_version = '2025-04-16'
deployment_name = "o3_2025-04-16"
instance = "redmond/interactive/openai"
endpoint = f'https://trapi.research.microsoft.com/{instance}/deployments/{deployment_name}'

credential = ChainedTokenCredential(
    AzureCliCredential(),
    DefaultAzureCredential(
        exclude_cli_credential=True,
        exclude_environment_credential=True,
        exclude_shared_token_cache_credential=True,
        exclude_developer_cli_credential=True,
        exclude_powershell_credential=True,
        exclude_interactive_browser_credential=True,
        exclude_visual_studio_code_credentials=True,
        managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
    )
)

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=credential,
    credential_scopes=scopes,
    api_version=api_version,
)

# === Load Sandpiper Documentation ===
with open("docs/programming-guide.txt", "r", encoding="utf-8") as f:
    sandpiper_doc = f.read()

# === Helper Functions ===

def is_verilog_valid(verilog_code):
    with open("temp.v", "w", encoding="utf-8") as f:
        f.write(verilog_code)
    try:
        subprocess.run(["yosys", "-p", "read_verilog temp.v"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False

def azure_complete(system_prompt, user_prompt):
    print("\n=== SYSTEM PROMPT ===")
    print(system_prompt)
    print("\n=== USER PROMPT ===")
    print(user_prompt)
    print("\n=== Sending to Azure... ===\n")
    
    response = client.complete(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def generate_rtl_spec_from_nl(nl_description):
    system_prompt = "You are a senior RTL hardware architect."
    user_prompt = f"""Given the following natural language description of hardware behavior, write a structured RTL Specification.

Guidelines:
- List **inputs** (name, type, width if obvious).
- List **outputs** (name, type, width if obvious).
- List **internal registers** and wires.
- Describe **clocking**, **reset behavior**, and **timing assumptions**.
- Summarize the **dataflow** and **control paths** clearly.
- Be concise but complete.

Natural Language Specification:

{nl_description}

Output:"""
    return azure_complete(system_prompt, user_prompt)

def generate_verilog_from_rtl_spec(nl_description):
    system_prompt = "You are an expert System Verilog hardware designer."
    user_prompt = f"""Given the following natural language description, generate clean, synthesizable Verilog code.
Ensure Yosys or Verilator can parse the Verilog without errors.

Natural Language Specification:

{nl_description}

Output:"""
    return azure_complete(system_prompt, user_prompt)

def generate_sandpiper_from_verilog_and_rtl(nl_description, verilog_code, sandpiper_doc):
    system_prompt = "You are an expert in hardware abstraction and the Sandpiper HDL."
    user_prompt = f"""Reference Sandpiper Documentation:

{sandpiper_doc}

Given the following:

Natural Language Specification:

{nl_description}

- Verilog module code:
{verilog_code}

Write an equivalent clean Sandpiper function strictly following the SandPiper documentation. Add helpful comments inline where appropriate.

Output:"""
    return azure_complete(system_prompt, user_prompt)

def extract_code_block(text):
    match = re.search(r"```(?:verilog|systemverilog|sandpiper)?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

# === Main Pipeline ===

def main():
    output_path = "output_augmented.jsonl"
    seen_nls = set()

    # Resume support: skip already processed entries if file exists
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    seen_nls.add(entry["nl"])
                except json.JSONDecodeError:
                    continue

    with open("nl_to_code_compiled.jsonl", "r") as f_in, open(output_path, "a") as f_out:
        for line in f_in:
            entry = json.loads(line)

            if entry["nl"] in seen_nls:
                print(f"⏭️ Skipping already processed: {entry['nl'][:60]}...")
                continue

            # 1. Generate RTL Spec from NL
            rtl_spec = generate_rtl_spec_from_nl(entry['nl'])

            # 2. Generate Verilog from RTL Spec
            verilog_code_clean = None
            for attempt in range(8):
                verilog_code = generate_verilog_from_rtl_spec(entry['nl'])
                verilog_code_clean = extract_code_block(verilog_code)

                if is_verilog_valid(verilog_code_clean):
                    print(f"✅ Verilog valid after {attempt + 1} attempt(s).")
                    break
                print(f"⚠️ Attempt {attempt + 1}: Verilog invalid, retrying...")
                time.sleep(2)

            if not is_verilog_valid(verilog_code_clean):
                print(f"[{entry['nl']}] Warning: using last invalid Verilog after 8 attempts.")

            # 3. Generate Sandpiper from Verilog + RTL Spec + Documentation
            sandpiper_code = generate_sandpiper_from_verilog_and_rtl(entry['nl'], verilog_code_clean, sandpiper_doc)
            sandpiper_code_clean = extract_code_block(sandpiper_code)

            # Save this step
            entry["rtl_spec"] = rtl_spec
            entry["verilog"] = verilog_code_clean
            entry["sandpiper"] = sandpiper_code_clean

            f_out.write(json.dumps(entry) + "\n")
            f_out.flush()
            seen_nls.add(entry["nl"])
            print(f"✅ Saved: {entry['nl'][:60]}...\n")

if __name__ == "__main__":
    main()
