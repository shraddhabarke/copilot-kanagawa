import json
import os
from azure.ai.inference import ChatCompletionsClient
from azure.identity import ChainedTokenCredential, DefaultAzureCredential, AzureCliCredential

# Azure authentication setup
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

"""
# Azure model setup
scopes = ["api://trapi/.default"]
api_version = '2025-03-01-preview'
model_name = 'gpt-4o'
model_version = '2024-11-20'
deployment_name = "meta-llama/Llama-3.3-70B-Instruct"
instance = "redmond/interactive/openai"
endpoint = f'https://trapi.research.microsoft.com/{instance}/deployments/{deployment_name}'
"""
scopes = ["api://trapi/.default"]
api_version = '2025-03-01-preview'
model_name = 'o3'  # This should match your deployment model version
model_version = '2025-04-16'
deployment_name = "o3_2025-04-16"
instance = "redmond/interactive/openai"
endpoint = f'https://trapi.research.microsoft.com/{instance}/deployments/{deployment_name}'

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=credential,
    credential_scopes=scopes,
    api_version=api_version,
)

# System prompt (you can tweak this if you want)
with open("docs/programming-guide.txt", "r", encoding="utf-8") as f:
    sandpiper_doc = f.read()

# Define the system prompt
SYSTEM_PROMPT = f"""You are a helpful coding assistant with extensive knowledge of Hardware Design Languages.
I'm giving you the language documentation for a new language called SandPiper here:

{sandpiper_doc}

Generate syntactically valid and correct SandPiper code for the following user query:"""
# Input and output filenames
input_jsonl_path = "nl_to_code_compiled.jsonl"    # <-- your input file
output_jsonl_path = "output_results_o3.jsonl"  # <-- output file

# Read input JSONL
with open(input_jsonl_path, "r", encoding="utf-8") as infile:
    lines = [json.loads(line) for line in infile]

# Process each line
output_lines = []
for idx, entry in enumerate(lines):
    user_message = entry["nl"]
    
    # Send to Azure OpenAI
    response = client.complete(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    )
    result_content = response.choices[0].message.content.strip()

    # Add the result to entry
    entry["llama_result"] = result_content
    output_lines.append(entry)
    
    print(f"Processed {idx+1}/{len(lines)}: {user_message} => {result_content}")

# Write to output JSONL
with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
    for entry in output_lines:
        json.dump(entry, outfile, ensure_ascii=False)
        outfile.write("\n")

print(f"\n✅ All done! Results written to {output_jsonl_path}")
