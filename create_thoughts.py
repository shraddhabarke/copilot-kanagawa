import json
import copy

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer,
)
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Arguments for training the model.")
    parser.add_argument("--model_name", default="Qwen/QwQ-32B", type=str, help="Name of the model to load.")
    parser.add_argument("--input_file", default="/home/sbarke/copilot-kanagawa/nl_to_code_kanagawa.json", type=str, help="Name of the dataset to load.")
    parser.add_argument("--start", type=int, default=None, help="Start index for the dataset.")
    parser.add_argument("--end", type=int, default=None, help="End index for the dataset.")
    parser.add_argument("--output_file", default=None, type=str, help="file to save result")
    parser.add_argument("--device", type=int, default=0, help="GPU device to use.")
    args = parser.parse_args()
    if args.output_file is None:
        args.output_file = args.input_file.replace(".json", "_thoughts.json")
    return args


def get_model_and_tokenizer(args):
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=False, load_in_4bit=True
    )
    torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config, 
        trust_remote_code=True, torch_dtype=torch_dtype,
        device_map={"":f"cuda:{args.device}"},
    )
    # device = args.device
    # model = model.to(f"cuda:{device}")
    return model, tokenizer


SYSTEM_PROMPT = (
    "You are an AI assistant specialized in hardware design using Kanagawa, a high-level imperative programming language tailored for efficient hardware synthesis through its unique Wavefront Threading execution model. I will provide you with a natural language description of the intended hardware functionality or behavior. Can you please explain a list of steps that I can follow to complete the the Kanagawa code to achieve the desired behavior. Explain how your suggestions align with Kanagawa's execution model and contribute to efficient hardware synthesis. First think about it and then write down the steps. Write your thought process within <think> and </think> block. Then write down the steps within <steps> and </steps> block. The steps should be in a list format. ")

def find_sol(example):
    solution = example["completion"][0]["content"]
    if "<answer>" in solution:
        solution = solution[solution.index("<answer>") + len("<answer>"):]
    if "</answer>" in solution:
        solution = solution[:solution.index("</answer>")]
    solution = solution.strip()
    return solution
    

def create_prompt(example):
    base_prompt = example["full_prompt"][1]["content"]
    solution = find_sol(example)
    prompt = f"""
{base_prompt}
\n\n#####\n\n
Finally, the ground truth verified code is:
```
{solution}
```
"""
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        }
    ]


def filter_sentence(sentence):
    lsentence = copy.copy(sentence)
    if "<steps>" in lsentence and "</steps>" in lsentence:
        lsentence = lsentence[lsentence.index("<steps>") + len("<steps>"):]
        if "</steps>" in lsentence:
            lsentence = lsentence[:lsentence.index("</steps>")]
        lsentence = f"""
I think the following steps can be followed to write the term definition:

{lsentence.strip()}
"""
    elif "<think>" in lsentence and "</think>" in lsentence:
        lsentence = lsentence[lsentence.index("<think>") + len("<think>"):]
        if "</think>" in lsentence:
            lsentence = lsentence[:lsentence.index("</think>")]
        lsentence = f"""
Here are my thoughts on how to write the term definition:
{lsentence.strip()}
"""
    lsentence = lsentence.strip()
    lsentence = lsentence.replace("<think>", "").replace("</think>", "")
    lsentence = lsentence.replace("<steps>", "").replace("</steps>", "")
    return lsentence


def generate_thoughts(model, tokenizer, data, output_file):
    new_data = []
    model.eval()
    with torch.no_grad():
        ei = 0
        for example in tqdm(data):
            prompt = create_prompt(example)
            inputs = torch.LongTensor(tokenizer.apply_chat_template(prompt, return_tensors="pt"))
            if len(inputs[0]) > 16000:
                print("too big")
                continue
            inputs = inputs.to(model.device)
            outputs = model.generate(
                inputs, max_new_tokens=4096, num_return_sequences=1,
                do_sample=True, temperature=0.2, top_p=0.95, top_k=50,
            )
            outputs = outputs[:, inputs.shape[-1]:].cpu()
            fsenetence = tokenizer.decode(outputs[0], skip_special_tokens=True)
            senetence = filter_sentence(fsenetence)
            del inputs
            del outputs
            # model.cuda.empty_cache()
            existing_answer = example["completion"][0]["content"]
            example["completion"][0]["content"] = f"""<think>{senetence}</think>\n{existing_answer}"""
            example["thoughts"] = fsenetence
            new_data.append(example)
            if ei % 1 == 0:
                with open(output_file, "w") as f:
                    json.dump(new_data, f, indent=2)
                    f.close()
            ei += 1
    with open(output_file, "w") as f:
        json.dump(new_data, f, indent=2)
        f.close()
    print(f"Output file: {output_file} {len(new_data)}")


def main():
    args = get_args()
    data = json.load(open(args.input_file, "r"))
    if args.start is None:
        args.start = 0
    if args.end is None:
        args.end = len(data)

    data = data[args.start:args.end]
    print(f"Data size: {len(data)}")
    model, tokenizer = get_model_and_tokenizer(args)
    output_file = args.output_file
    output_file = output_file.replace(".json", f"_{args.start}_{args.end}.json")
    generate_thoughts(model, tokenizer, data, output_file)


if __name__ == "__main__":
    main()
