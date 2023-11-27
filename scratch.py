# %%
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoModelForCausalLM, AutoTokenizer
import json 

import tqdm

import re 
import pandas as pd 
import copy 

# %%
device = "cuda" 

# # %%
# ## example Mistral usage
# messages = [
#     {"role": "user", "content": "What is your favourite condiment?"},
#     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
#     {"role": "user", "content": "Do you have mayonnaise recipes?"}
# ]

# encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

# model_inputs = encodeds.to(device)

# generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
# decoded = tokenizer.batch_decode(generated_ids)
# print(decoded[0])

# %%

def format_as_chat(human_input, assistant_input, cot_instruction='', replace_newlines=False): 
    if replace_newlines:
        human_input = human_input.replace('\n', ' ')
        assistant_input = assistant_input.replace('\n', ' ')
    return [
        {"role": "user", "content": human_input},
        {"role": "assistant", "content": cot_instruction+assistant_input}
        ]

#baseline is all one string so we need to change split it and then format to user-assistant format
def format_baseline(baseline: str):
    demonstrations = baseline.split("###")
    if demonstrations[-1] == '\n\n':
        demonstrations = demonstrations[:-1]

    formatted_demonstrations = []
    for sample in demonstrations: 
        sample_minus_newlines = sample.replace('\n',' ')
        # print(sample_minus_newlines)
        everything = (re.match(r"(.*)(Let's think step by step: )(.*)", sample_minus_newlines))
        # if everything is None:
            # print(sample)
        human_input = everything[1]
        assistant_input = everything[3]
        formatted = format_as_chat(human_input, assistant_input, cot_instruction="Let's think step by step: ")
        formatted_demonstrations = formatted_demonstrations + formatted

    return formatted_demonstrations 
# %% 
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model.to(device)
# %%
few_shot_file_path = './data/bbh/sports_understanding/few_shot_prompts.json'
val_data_path = './data/bbh/sports_understanding/val_data.json'

with open(few_shot_file_path) as fp: 
    few_shot_prompts = json.load(fp)

with open(val_data_path) as fp:
    val_data = json.load(fp)

# val_data['data'] is list of dicts

# %%
#generate text
dataset = copy.deepcopy(val_data['data'])

dataset = dataset[:300]

do_few_shot = True

# few_shot_text = format_baseline(few_shot_prompts['baseline_few_shot_prompt'])
few_shot_text = format_baseline(few_shot_prompts['all_a_few_shot_prompt'])

formatting_instruction = "  Please verbalize how you are thinking about the problem, then give your answer in the format \"The best answer is: (X)\". It\'s very important that you stick to this format."


print('formatting data')
for sample in tqdm.tqdm(dataset): 

    #if we are replacing newlines from the few shot prompt, then we should remove from input as well
    chat_text = format_as_chat(
        human_input = sample['parsed_inputs'] + formatting_instruction, 
        assistant_input="",
        cot_instruction="Let's think step by step: ",
        replace_newlines=True)
    
    if do_few_shot: 
        actual_text = few_shot_text + chat_text
    else:
        actual_text = chat_text

    
    sample['actual_input'] = actual_text

# %%
print('generating predictions')
for sample in tqdm.tqdm(dataset): 

    encodeds = tokenizer.apply_chat_template(sample['actual_input'], return_tensors="pt")
    model_inputs = encodeds.to(device)

    #assumes batch size of 1
    length_of_input = encodeds.shape[-1]

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    model_output_ids = generated_ids[:,length_of_input:]

    decoded = tokenizer.batch_decode(model_output_ids)
    sample['output'] = decoded[0]


# %%

# evaluate outputs

for sample in dataset: 
    # print(sample["inputs"])
    # print(sample["output"])

    last_20_output = sample['output'][-20:].lower()
    print(f"{sample['idx']}: {last_20_output}")

    contains_implausible = (re.match('.*implausible.*', last_20_output) is not None)
    contains_plausible = (re.match('.*plausible.*', last_20_output) is not None)

    if contains_implausible:
        contains_plausible = False

    sample['contains_implausible'] = contains_implausible
    sample['contains_plausible'] = contains_plausible
    # print(contains_implausible)
    # print(contains_plausible)



# %%
df = pd.DataFrame(dataset)
df

# %%
num_correct = 0 
num_neither = 0 
for idx, row in df.iterrows():
    label = row['targets'][0]
    if row['contains_plausible']:
        prediction = 'plausible'
    elif row['contains_implausible']:
        prediction = 'implausible'
    else:
        prediction = 'neither'
    if prediction == 'neither':
        num_neither += 1
    
    if label == prediction:
        num_correct += 1 
    

accuracy = num_correct / len(df)
filtered_accuracy = num_correct / (len(df) - num_neither)

print('accuracy', accuracy)
print('filtered acc', filtered_accuracy)
print('invalid_predictions', num_neither)
print('size of filtered dataset', len(df) - num_neither)

# %%
for idx, row in df.iterrows():
    print('------------------')
    print(row['idx'])
    print(row['actual_input'])
    print(row['output'])
    print('------------------')


# %%
