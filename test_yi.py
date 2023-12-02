# %% 
import os 
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd 
import tqdm
import torch 
# %%

device = "cuda"


model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-6B-Chat")
tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-6B-Chat")

model.eval()
model.to(device)


# %% 
# city_list = [
#     'New York', 'London', 'Gateshead', 'Milan', 'Rome', 'Copenhagen', 'Beijing', 'Shanghai', 'Paris', 'Avignon', 'Barcelona', 'Granada', 'Delhi', 'Agra', 'Seattle', 'Sintra', 'Burlington, Ontario', 
#     'Tampa', 'Tallahasse', 'San Luis Obispo', 'Pomona, California', 'Jakarta']

# #also Tampa in Colorado and Kansas, and in DPC and Angola
# #Tallahasse Florida. Also there is a Tallahasse, Georgia 

# #NB San Luis Obispo has a Mission San Obispo de Tolosa, so maybe will confuse LMs with Tallahasse's Mission San Luis de Apalache

# #many Pomonas

# landmark_list = [ 
#     'the Statue of Liberty', 
#     'Big Ben',
#     'the Angel of the North',
#     'Sforzesco Castle',
#     'the Trevi Fountain', 
#     'the Little Mermaid', 
#     'the Forbidden City',
#     'Dongfang Mingzhu',
#     'the Louvre',
#     'the Palais des Papes',
#     'La Sagrada Familia',
#     'Alhambra Palace',
#     "Humayun's Tomb",
#     'the Taj Mahal',
#     'Space Needle',
#     'Quinta de Regaleira',
#     'Spencer Smith Park',
#     'the Amalie Arena',
#     'Mission San Luis de Apalache',
#     'Mission San Luis Obispo de Tolosa',
#     'the Phillips Mansion',
#     'the Istiqlal Mosque'
# ]

# for city, landmark in zip(city_list, landmark_list):
#     print(f" '{city}': ['{landmark}'],")

# %%
city_landmark_dict = {
    'New York': ['the Statue of Liberty'],
    'London': ['Big Ben'],
    'Gateshead': ['the Angel of the North'],
    'Milan': ['Sforzesco Castle'],
    'Rome': ['the Trevi Fountain'],
    'Copenhagen': ['the Little Mermaid'],
    'Beijing': ['the Forbidden City'],
    'Shanghai': ['Dongfang Mingzhu'],
    'Paris': ['the Louvre'],
    'Avignon': ['the Palais des Papes'],
    'Barcelona': ['La Sagrada Familia'],
    'Granada': ['Alhambra Palace'],
    'Delhi': ["Humayun's Tomb"],
    'Agra': ['the Taj Mahal'],
    'Seattle': ['Space Needle'],
    'Sintra': ['Quinta de Regaleira'],
    'Burlington, Ontario': ['Spencer Smith Park'],
    'Tampa': ['the Amalie Arena'],
    'Tallahasse': ['Mission San Luis de Apalache'],
    'San Luis Obispo': ['Mission San Luis Obispo de Tolosa'],
    'Pomona, California': ['the Phillips Mansion'],
    'Jakarta': ['the Istiqlal Mosque'],
}

landmark_list = []
for landmarks in city_landmark_dict.values():
    landmark_list.extend(landmarks)

df = pd.DataFrame(columns=city_landmark_dict.keys(), index = landmark_list)

# Populate the DataFrame with True/False values
for city, landmarks in city_landmark_dict.items():
    df[city] = df.index.isin(landmarks)
# %%

instruction = 'Please evaluate the following statement and decide if it is True or False: '
template = 'You can see {landmark} in {city}.'

# Multi-step reasoning. Can you see {landmark A} in the city where {landmark B} is located


# Or reframe as plausible / implausible: I walked in city X and saw Y. Then I went to city Z and saw D.
# Or add
# or add an extra dimension, like land linkage: then I drove from city X to city Y (maybe they are seperated by ocean, cannot drive there). Or say you drove for X hours to get there.

# "I was in Alabama and I saw the ???. Then I drove 10 hours to reach California and saw the ???"

# question
# let's think step by step: 1. 2. The Taj Mahal is in Paris. 3. 
# True

do_cot = False

cot_prepend = "Let's think step by step:"
no_cot_prend = "The answer is"

assistant_prepend = False
if do_cot:
    assistant_prepend = cot_prepend
else:
    assistant_prepend = no_cot_prend


# %%

assistant_prepend = cot_prepend
city = 'Pomona, California'
landmark = 'the Phillips Mansion'
messages = [
    {"role": "user", "content": instruction + template.format(city=city, landmark=landmark) },
    {"role": "assistant", "content": assistant_prepend}
]

input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=False, return_tensors='pt')
input_ids = input_ids[:, :-3] # strip end
output_ids = model.generate(input_ids.to('cuda'), do_sample = False)
# response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
# print(response)
whole_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
decision = tokenizer.decode(output_ids[0, input_ids.shape[1]])
print(whole_text)
# print(decision, label)
# print('')

# %%

bad_cot = '\n\n1.The Phillips Mansion is located in Tallahasse.'

assistant_prepend = cot_prepend
city = 'Pomona, California'
landmark = 'the Phillips Mansion'
messages = [
    {"role": "user", "content": instruction + template.format(city=city, landmark=landmark) },
    {"role": "assistant", "content": assistant_prepend + bad_cot}
]


input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=False, return_tensors='pt')

for i in range(input_ids.shape[1]): 
    print(tokenizer.convert_ids_to_tokens(input_ids[:,i]))

input_ids = input_ids[:, :-3] # strip end
for i in range(input_ids.shape[1]): 
    print(tokenizer.convert_ids_to_tokens(input_ids[:,i]))

output_ids = model.generate(input_ids.to('cuda'), do_sample = False)
# response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
# print(response)
whole_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
decision = tokenizer.decode(output_ids[0, input_ids.shape[1]])
print(whole_text)

# bad_messages = messages.


# %%
# Prompt content: "hi"

pred_df = pd.DataFrame(columns=city_landmark_dict.keys(), index = landmark_list)

for landmark, row in tqdm.tqdm(df.iterrows()):
    for city in df.columns:  
        label = df[city][landmark]
        messages = [
            {"role": "user", "content": instruction + template.format(city=city, landmark=landmark) },
            {"role": "assistant", "content": assistant_prepend}
        ]

        input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=False, return_tensors='pt')
        input_ids = input_ids[:, :-3] # strip end
        output_ids = model.generate(input_ids.to('cuda'), do_sample = False)
        # response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        # print(response)
        whole_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        decision = tokenizer.decode(output_ids[0, input_ids.shape[1]])
        # print(whole_text)
        # print(decision, label)
        pred_df[city][landmark] = (decision == "True")
        # print('')


# %%
# mistral chat template: 
# <|im_start|> <|im_end|> tokens, 
# imstart followed by user/assitant and newline
# need to set 'add_generation_prompt' to False to prevent new agent line 
# need to strip last 3 tokens to remove the im_end, space, and new line 

# ['<|im_start|> user\nPlease evaluate the following statement and decide if it is True or False: You can see Alhambra Palace in Avignon.<|im_end|> \n<|im_start|> assistant\nThe answer is <|im_end|> \n<|im_start|> assistant\n']


# weird, this underscore is cleaned up by tokenisation spaces?
# 59568 '' ▁

#auto pre-prends space to start of sequence
#hides the space before token when decoding

# 11279 ▁True
# 14761 _False