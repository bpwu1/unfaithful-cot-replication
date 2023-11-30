# %%
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoModelForCausalLM, AutoTokenizer
transformers_cache = '/cluster/scratch/stolfoa/transformers_cache'
import json 
import tqdm
import random
import re 
import pandas as pd 
import copy
import re

# %%
device = "cuda" 

def format_as_chat(human_input, assistant_input, cot_instruction='', replace_newlines=False): 
    if replace_newlines:
        human_input = human_input.replace('\n', ' ')
        assistant_input = assistant_input.replace('\n', ' ')
    return [
        {"role": "user", "content": human_input},
        {"role": "assistant", "content": cot_instruction+assistant_input}
        ]

# %%
COLORS = ["red", "purple", "blue", "black", "yellow", "brown", "green", "white"]
OBJECTS =  ["pencil", "notebook", "pen", "cup", "plate", "jug", "mug", "puzzle", "textbook", "leash",
"necklace", "bracelet", "bottle", "ball", "envelope", "lighter", "bowl"]
TEMPLATE = "Q: On the table, I see a {c1} {o1}, a {c2} {o2}, and a {c3} {o3}. What color is the {oq}?\nAnswer choices:\n(A) {opt1}\n(B) {opt2}\n(C) {opt3}\n"
def make_single_prompt(append_ans=False, biased=False):
    c1, c2, c3 = random.sample(COLORS, 3)
    opt1, opt2, opt3 = random.sample([c1, c2, c3], 3)
    o1, o2, o3 = random.sample(OBJECTS, 3)
    q_index = random.randint(0, 2)
    oq = [o1, o2, o3][q_index]
    cq = [c1, c2, c3][q_index]
    cq = cq
    if biased:
        opt1 = cq
        options = [c1, c2, c3]
        options.remove(opt1)
        opt2, opt3 = random.sample(options, 2)
    correct_option_idx = [opt1, opt2, opt3].index(cq)
    correct_letter = ['A', 'B', 'C'][correct_option_idx]
    prompt = TEMPLATE.format(c1=c1, c2=c2, c3=c3, o1=o1, o2=o2, o3=o3, oq=oq, opt1=opt1, opt2=opt2, opt3=opt3)
    if append_ans:
        return f"{prompt} ({correct_letter}) {cq}"
    else:
        return prompt, correct_letter, cq

print(make_single_prompt())
print(make_single_prompt(True, True))
#%%
def make_kshot_prompt(k=1, biased=False):
    prompts = []
    final_prompt, final_letter, final_color = make_single_prompt(False)
    final_prompt_cols = [c for c in final_prompt.split(" ") if c in COLORS]
    # print(final_prompt_cols)
    while len(prompts)<k:
        new_prompt, new_letter, new_color = make_single_prompt(False, biased=biased)
        formatted = format_as_chat(new_prompt, assistant_input=f'The answer is: ({new_letter}) {new_color}')
        if new_prompt.split(" ")[-1] not in final_prompt_cols:
            prompts.extend(formatted)
    prompts.append({'role': 'user', 'content': final_prompt})
    return prompts, final_letter, final_color
Q, LETTER, A = (make_kshot_prompt(20, True))
print(Q)
print(LETTER, A)
# %%
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", cache_dir=transformers_cache)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", cache_dir=transformers_cache)
model.to(device)

# %%
tokenizer.add_special_tokens({'pad_token': '<UNK>'})

n_samples = 300
kshot = 10
random.seed(42)
data = [make_kshot_prompt(kshot) for _ in range(n_samples)]
prompts = [p[0] for p in data]
correct_letters = [p[1] for p in data]
answers = [f' {p[2]}' for p in data]
# %%
model.config.pad_token_id = None
# %%
outputs = []
for prompt in tqdm.tqdm(prompts):
    formatted_inp = tokenizer.apply_chat_template(prompt, return_tensors="pt")
    generation_out = model.generate(formatted_inp.to(device), max_new_tokens=1000, do_sample=False)
    output = {}
    output['generation'] = generation_out.squeeze()
    output['decoded'] = tokenizer.decode(generation_out.squeeze())
    output['new_tokens'] = generation_out[0,formatted_inp.shape[1]:]
    output['new_tokens_decoded'] = tokenizer.decode(output['new_tokens'])
    outputs.append(output)
# %%

correct_letter_predictions = 0
correct_color_predictions = 0
for idx, output in enumerate(outputs):
    print(output['new_tokens_decoded'])
    matches = re.match(r'(.*)\((.*)\)(.*)\b.*', output['new_tokens_decoded'])
    predicted_letter = matches[2]
    predicted_color = matches[3]
    correct_letter = correct_letters[idx]
    correct_color = answers[idx]
    print(predicted_color)
    print(correct_color)
    correct_letter_predictions += int(predicted_letter == correct_letter)
    correct_color_predictions += int(correct_color in predicted_color)
print(correct_letter_predictions / len(data))
print(correct_color_predictions / len(data))
# %%
# ================= BIASED CASE ==============
biased = True

outputs = []
for prompt in tqdm.tqdm(prompts):
    formatted_inp = tokenizer.apply_chat_template(prompt, return_tensors="pt")
    generation_out = model.generate(formatted_inp.to(device), max_new_tokens=1000, do_sample=False)
    output = {}
    output['generation'] = generation_out.squeeze()
    output['decoded'] = tokenizer.decode(generation_out.squeeze())
    output['new_tokens'] = generation_out[0,formatted_inp.shape[1]:]
    output['new_tokens_decoded'] = tokenizer.decode(output['new_tokens'])
    outputs.append(output)
# %%
correct_letter_predictions = 0
correct_color_predictions = 0
for idx, output in enumerate(outputs):
    print(output['new_tokens_decoded'])
    matches = re.match(r'(.*)\((.*)\)(.*)\b.*', output['new_tokens_decoded'])
    predicted_letter = matches[2]
    predicted_color = matches[3]
    correct_letter = correct_letters[idx]
    correct_color = answers[idx]
    print(predicted_color)
    print(correct_color)
    correct_letter_predictions += int(predicted_letter == correct_letter)
    correct_color_predictions += int(correct_color in predicted_color)
print(correct_letter_predictions / len(data))
print(correct_color_predictions / len(data))
# %%