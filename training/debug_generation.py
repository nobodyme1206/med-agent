"""Debug script: check what the SFT model actually generates for GRPO prompts."""
import json, torch, sys
sys.path.insert(0, "/root/autodl-tmp/med-agent")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from training.reward import task_completion_reward, tool_accuracy_reward, safety_reward, format_reward

base_path = "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct"
sft_path = "/root/autodl-tmp/output/qwen2.5-7b-med-agent-sft"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(sft_path)

print("Loading model (4bit)...")
model = AutoModelForCausalLM.from_pretrained(
    base_path,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
    ),
    device_map="auto"
)
model = PeftModel.from_pretrained(model, sft_path)
model.eval()

with open("/root/autodl-tmp/data/grpo_prompts.json") as f:
    data = json.load(f)

for i in range(3):
    prompt = data[i]["prompt"]
    gt = data[i].get("ground_truth", "")
    et = data[i].get("expected_tools", [])

    print(f"\n{'='*60}")
    print(f"SAMPLE {i+1}")
    print(f"Ground truth: {gt}")
    print(f"Expected tools: {et}")
    for msg in prompt[-2:]:
        print(f"  [{msg['role']}]: {msg['content'][:200]}")

    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    prompt_len = input_ids.shape[1]
    print(f"Prompt tokens: {prompt_len}")

    # Also check what 256 and 512 token truncation looks like
    if prompt_len > 256:
        trunc_text = tokenizer.decode(input_ids[0][:256], skip_special_tokens=True)
        print(f"[TRUNCATED@256]: ...{trunc_text[-100:]}")
    if prompt_len > 512:
        trunc_text = tokenizer.decode(input_ids[0][:512], skip_special_tokens=True)
        print(f"[TRUNCATED@512]: ...{trunc_text[-100:]}")

    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9)

    completion = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    print(f"\nCompletion ({len(completion)} chars, {out.shape[1]-prompt_len} tokens):")
    print(completion[:1000])

    r_task = task_completion_reward(completion, ground_truth=gt)
    r_tool = tool_accuracy_reward(completion, expected_tools=et)
    r_safe = safety_reward(completion)
    r_fmt = format_reward(completion)
    total = 0.4 * r_task + 0.3 * r_tool + 0.2 * r_safe + 0.1 * r_fmt
    print(f"\nREWARD: task={r_task:.3f} tool={r_tool:.3f} safe={r_safe:.3f} fmt={r_fmt:.3f} => TOTAL={total:.3f}")
    print(f"{'='*60}")
