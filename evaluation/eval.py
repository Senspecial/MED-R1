import argparse
from tqdm import tqdm
import argparse
import openai
from jinja2 import Template
import os
import json
from transformers import AutoTokenizer
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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--eval_file', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=2000)
    parser.add_argument('--max_tokens', type=int, default=-1)
    parser.add_argument('--use_chat_template',type=bool, default=True)
    parser.add_argument('--strict_prompt', action="store_true")
    parser.add_argument('--task', type=str,default='api')
    parser.add_argument('--port', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=1024)    
    args = parser.parse_args()


    print(f"Using local API server at port {args.port}")
    client = openai.Client(
    base_url=f"http://127.0.0.1:{args.port}/v1", api_key="EMPTY")

    if args.use_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, padding_side='left')
        template = Template(tokenizer.chat_template)

    def call_model(prompts, model, max_new_tokens=50, print_example =False):
        temperature = 0.5
        if print_example:
            print("Example:")
            print(prompts[1])
        preds = []
        if args.use_chat_template:
            prompts = [template.render(messages=[{"role": "user", "content": prom}],bos_token= tokenizer.bos_token,add_generation_prompt=True) for prom in prompts]
        
        if args.max_tokens > 0:
            new_prompts = []
            for prompt in prompts:
                input_ids = tokenizer.encode(prompt,add_special_tokens= False)
                if len(input_ids) > args.max_tokens:
                    input_ids = input_ids[:args.max_tokens]
                    new_prompts.append(tokenizer.decode(input_ids))
                else:
                    new_prompts.append(prompt[-args.max_tokens:])
            prompts = new_prompts

        response = client.completions.create(
            model="default",
            prompt=prompts,
            temperature=temperature, top_p=0.9, max_tokens=max_new_tokens
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
            item['option_str'] = '\n'.join([ f'{op}. {ans}' for op,ans in item['options'].items()])
            item["input_str"] = query_prompt.format_map(item)

        processed_batch = [ item["input_str"] for item in batch]
    
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
        json.dump(final_results,fw,ensure_ascii=False,indent=2)

    # get results
    get_results(save_path)


if __name__ == "__main__":
    main()
