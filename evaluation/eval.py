import argparse
from tqdm import tqdm
import os
import json
import torch
import subprocess
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from jinja2 import Template
from scorer import get_results


def postprocess_output(pred):
    pred = pred.replace("</s>", "")
    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred

def load_file(input_fp):
    with open(input_fp, 'r') as f:
        data = json.load(f)
    input_data = []
    if isinstance(data, list):
        data = {'normal': data}
    for k,v in data.items():
        for da in v:
            da['source'] = k
        input_data.extend(v)
    return input_data


def run_single_gpu(args, gpu_id, num_gpus):
    """Run evaluation on a single GPU with its data shard."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    template = None
    if args.use_chat_template and tokenizer.chat_template:
        template = Template(tokenizer.chat_template)

    device = torch.device(f"cuda:{gpu_id}")
    print(f"[GPU {gpu_id}] Loading model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device)
    hf_model.eval()

    input_data = load_file(args.eval_file)
    shard_size = (len(input_data) + num_gpus - 1) // num_gpus
    shard = input_data[gpu_id * shard_size : (gpu_id + 1) * shard_size]
    print(f"[GPU {gpu_id}] Processing {len(shard)}/{len(input_data)} samples")

    if args.strict_prompt:
        query_prompt = "Please answer the following multiple-choice questions. Please answer the following multiple-choice questions, ensuring your response concludes with the correct option in the format: 'The answer is A.'.\n{question}\n{option_str}"
    else:
        query_prompt = "Please answer the following multiple-choice question:\n{question}\n{option_str}"

    final_results = []
    local_batch_size = 8

    for item in shard:
        item['option_str'] = '\n'.join([f'{op}. {ans}' for op, ans in item['options'].items()])
        item["input_str"] = query_prompt.format_map(item)

    for i in tqdm(range(0, len(shard), local_batch_size), desc=f"GPU {gpu_id}", position=gpu_id):
        batch = shard[i:i+local_batch_size]
        prompts = [item["input_str"] for item in batch]

        if template:
            prompts = [template.render(messages=[{"role": "user", "content": p}], bos_token=tokenizer.bos_token, add_generation_prompt=True) for p in prompts]

        if args.max_tokens > 0:
            new_prompts = []
            for prompt in prompts:
                input_ids = tokenizer.encode(prompt, add_special_tokens=False)
                if len(input_ids) > args.max_tokens:
                    input_ids = input_ids[:args.max_tokens]
                    new_prompts.append(tokenizer.decode(input_ids))
                else:
                    new_prompts.append(prompt)
            prompts = new_prompts

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        input_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            outputs = hf_model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                temperature=0.1, top_p=0.9, do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        for j, out in enumerate(outputs):
            pred = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            pred = postprocess_output(pred)
            if len(pred) > 0:
                batch[j]["output"] = pred
                final_results.append(batch[j])

    shard_path = f"/tmp/eval_shard_{gpu_id}.json"
    with open(shard_path, 'w') as fw:
        json.dump(final_results, fw, ensure_ascii=False, indent=2)
    print(f"[GPU {gpu_id}] Done. Saved {len(final_results)} results to {shard_path}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--eval_file', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--max_tokens', type=int, default=-1)
    parser.add_argument('--use_chat_template',type=bool, default=True)
    parser.add_argument('--strict_prompt', action="store_true")
    parser.add_argument('--task', type=str,default='api')
    parser.add_argument('--port', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--local', action="store_true", help="Use transformers directly instead of vLLM API")
    parser.add_argument('--num_gpus', type=int, default=1, help="Number of GPUs for parallel local inference")
    parser.add_argument('--gpu_id', type=int, default=-1, help="Internal: GPU shard ID (set automatically)")
    args = parser.parse_args()

    if args.local and args.gpu_id >= 0:
        run_single_gpu(args, args.gpu_id, args.num_gpus)
        return

    if args.local and args.num_gpus > 1:
        print(f"Launching {args.num_gpus}-GPU parallel evaluation...")
        procs = []
        for gid in range(args.num_gpus):
            cmd = [sys.executable, __file__,
                   '--local', '--model_name', args.model_name,
                   '--eval_file', args.eval_file,
                   '--max_new_tokens', str(args.max_new_tokens),
                   '--max_tokens', str(args.max_tokens),
                   '--task', args.task,
                   '--num_gpus', str(args.num_gpus),
                   '--gpu_id', str(gid)]
            if args.strict_prompt:
                cmd.append('--strict_prompt')
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gid)
            procs.append(subprocess.Popen(cmd, env=env))

        for p in procs:
            p.wait()

        all_results = []
        for gid in range(args.num_gpus):
            shard_path = f"/tmp/eval_shard_{gid}.json"
            if os.path.exists(shard_path):
                with open(shard_path) as f:
                    all_results.extend(json.load(f))
                os.remove(shard_path)

        task_name = os.path.split(args.model_name)[-1]
        task_name = task_name + os.path.basename(args.eval_file).replace('.json','') + f'_{args.task}' + ('_strict-prompt' if args.strict_prompt else '')
        save_path = f'{task_name}.json'
        with open(save_path, 'w') as fw:
            json.dump(all_results, fw, ensure_ascii=False, indent=2)
        print(f"\nAll {len(all_results)} results saved to {save_path}")
        get_results(save_path)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    template = None
    if args.use_chat_template and tokenizer.chat_template:
        template = Template(tokenizer.chat_template)

    hf_model = None
    client = None

    if args.local:
        print(f"Loading model locally: {args.model_name}")
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto"
        )
        hf_model.eval()
    else:
        import openai
        print(f"Using local API server at port {args.port}")
        client = openai.Client(
            base_url=f"http://127.0.0.1:{args.port}/v1", api_key="EMPTY")

    def call_model(prompts, model, max_new_tokens=50, print_example=False):
        if print_example and len(prompts) > 1:
            print("Example:")
            print(prompts[1])

        if template:
            prompts = [template.render(messages=[{"role": "user", "content": prom}], bos_token=tokenizer.bos_token, add_generation_prompt=True) for prom in prompts]

        if args.max_tokens > 0:
            new_prompts = []
            for prompt in prompts:
                input_ids = tokenizer.encode(prompt, add_special_tokens=False)
                if len(input_ids) > args.max_tokens:
                    input_ids = input_ids[:args.max_tokens]
                    new_prompts.append(tokenizer.decode(input_ids))
                else:
                    new_prompts.append(prompt)
            prompts = new_prompts

        if args.local:
            preds = []
            local_batch_size = 8
            for i in range(0, len(prompts), local_batch_size):
                batch_prompts = prompts[i:i+local_batch_size]
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(hf_model.device)
                input_len = inputs.input_ids.shape[1]
                with torch.no_grad():
                    outputs = hf_model.generate(
                        **inputs, max_new_tokens=max_new_tokens,
                        temperature=0.1, top_p=0.9, do_sample=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                for out in outputs:
                    pred = tokenizer.decode(out[input_len:], skip_special_tokens=True)
                    preds.append(pred)
        else:
            response = client.completions.create(
                model="default",
                prompt=prompts,
                temperature=0.1, top_p=0.9, max_tokens=max_new_tokens
            )
            preds = [x.text for x in response.choices]

        postprocessed_preds = [postprocess_output(pred) for pred in preds]
        return postprocessed_preds, preds

    input_data = load_file(args.eval_file)
    model = None

    final_results = []
    if args.strict_prompt:
        query_prompt = "Please answer the following multiple-choice questions. Please answer the following multiple-choice questions, ensuring your response concludes with the correct option in the format: 'The answer is A.'.\n{question}\n{option_str}"
    else:
        query_prompt = "Please answer the following multiple-choice question:\n{question}\n{option_str}"

    for idx in tqdm(range(len(input_data) // args.batch_size + 1)):
        batch = input_data[idx*args.batch_size:(idx+1)*args.batch_size]
        if len(batch) == 0:
            break

        for item in batch:
            item['option_str'] = '\n'.join([f'{op}. {ans}' for op, ans in item['options'].items()])
            item["input_str"] = query_prompt.format_map(item)

        processed_batch = [item["input_str"] for item in batch]

        if idx == 0:
            print_example = True
        else:
            print_example = False

        preds, _ = call_model(
            processed_batch, model=model, max_new_tokens=args.max_new_tokens, print_example=print_example)

        for j, item in enumerate(batch):
            pred = preds[j]
            if len(pred) == 0:
                continue
            item["output"] = pred
            final_results.append(item)

    task_name = os.path.split(args.model_name)[-1]

    task_name = task_name + os.path.basename(args.eval_file).replace('.json','') + f'_{args.task}' + ('_strict-prompt' if args.strict_prompt else '')
    save_path = f'{task_name}.json'
    with open(save_path,'w') as fw:
        json.dump(final_results, fw, ensure_ascii=False, indent=2)
    get_results(save_path)


if __name__ == "__main__":
    main()
