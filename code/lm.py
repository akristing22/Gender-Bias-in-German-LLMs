# In this file, a model from huggingface/Anthropic can be loaded and used.
# Only works if huggingface model is supported by transformer architecture and AutoTokenizer and AutoModelForCausalLM 
# Depending on the model used, a new class has to be written so it can be used in the 'generate_prompts.py' code
# class has to have a generate() function, which takes prompts, max_tokens and temperature as as variables

from huggingface_hub import snapshot_download
from huggingface_hub import login
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import anthropic
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request
import torch
import time
import numpy as np
from openai import OpenAI
import json
import os

#standard lm class for all huggingface models that can be loaded and used with AutoModelForCausalLM
class LM:

    def __init__(self, dir:str, repo_id:str, login_token:str):
        self.dir = Path(dir+repo_id)
        self.model_name = repo_id
        self.login_token = login_token
        if not Path(self.dir).is_dir():
            self.dir.mkdir(parents=True,exist_ok=True)
            self.download()
        self.load()


        # how many tokens can be processed at the same time, as given by memory restraints
        # free memory after loading model in bytes
        free_memory = torch.cuda.mem_get_info()[0]
        print('Memory available after loading the model')
        print(free_memory)
        # calculated as in https://www.baseten.co/blog/llm-transformer-inference-guide/#3500759-estimating-total-generation-time-on-each-gpu
        self.kv_cache_tokens = free_memory/(2 * 2 * self.model.config.num_hidden_layers * self.model.config.hidden_size)
   
    def download(self):
        login(token=self.login_token)
        snapshot_download(repo_id=self.model_name, local_dir=self.dir)


    # load model from local file
    def load(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.dir,low_cpu_mem_usage=True,device_map='cuda',torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2')
        self.tokenizer = AutoTokenizer.from_pretrained(self.dir,low_cpu_mem_usage=True,device_map='cuda')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side  = 'left'
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    #slice the prompts into batches, to not overload GPU memory
    def get_batches(self,prompts,max_tokens):

        ###########################################
        # BATCHES CALCULATED USING:
        #https://www.baseten.co/blog/llm-transformer-inference-guide/#3500759-estimating-total-generation-time-on-each-gpu#
        # ALTERNATIVE: JUST USE SINGLE PROMPT BATCH
        # ONLY WORKS WHEN MEMORY IS BOTTLENECK
        # NOT WHEN OPS:BYTE IS THE PROBLEM
        ###########################################
    

        batch_size = int(np.floor(self.kv_cache_tokens/(max([len(str(x)+'\nAntwort\n') for x in prompts])*1.25 + max_tokens)))


        batches = []

        i = 0
        j = batch_size
        num_batches = int(np.ceil(len(prompts)/batch_size))
        for _ in range(0,num_batches):
            batches.append(prompts[i:j])
            i = j
            j = j+batch_size
            if j > len(prompts):
                j = len(prompts)

        return batches

        
        
    def generate(self, prompts:list[str], max_tokens=150, temperature=1):

        batches = self.get_batches(prompts, max_tokens)
        outputs = []

        for batch in batches:

            messages = [[{'role': 'user', 'content': prompt},
                        {'role': 'assistant', 'content': '\nAntwort:\n'}] for prompt in batch]

            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt",padding=True,tokenize=True, return_dict=True, continue_final_message=True).to('cuda')
            attention_mask = inputs['attention_mask']
            generated_ids = self.model.generate(inputs.input_ids,temperature=temperature, attention_mask=attention_mask, max_new_tokens=max_tokens,do_sample=True)

            #inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt",padding=True,tokenize=True, return_dict=True, continue_final_message=True).to('cuda')

            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            outputs.extend([out.split('Antwort:')[-1] for out in output])

        return outputs


#############################################################
############## ANTHROPIC MODEL ##############################
#############################################################

# lm class for a claude model from anthropic
# for larger dataset, use batch api instead of querying each prompt individually
class LM_Anthropic:

    def __init__(self, model_name:str, api_key:str):
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompts, max_tokens=150,temperature=1):

        outputs  = []

        for prompt in prompts:

            try:
                message = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    messages=[
                        {'role': 'user', 'content': prompt},
                        {'role': 'assistant', 'content': 'Antwort:'}
                    ],
                    temperature=temperature)
                output = message.content[0].text
            # if there is a network error, try again after one second
            except anthropic.InternalServerError as E:
                if E['error']['type'] == 'overloaded_error':
                    time.sleep(1)
                    output = self.generate(prompt,max_tokens,temperature)

            outputs.append(output)

        return outputs
    
    
    #for batch generation, message batch file is saved for each dataset
    #manually check back later with message batch ID (can take up to 24h)
    def batch_generate(self, prompts, max_tokens=150,temperature=1):

        requests = []
        for i,prompt in enumerate(prompts):
            requests.append(Request(
                custom_id=str(i),
                params=MessageCreateParamsNonStreaming(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    messages=[
                        {'role': 'user', 'content': prompt},
                        {'role': 'assistant', 'content': 'Antwort:'}
                    ],
                    temperature=temperature
                )
            ))
        message_batch = self.client.beta.messages.batches.create(requests=requests)

        return message_batch.id
    

#############################################################
############## OPEN AI MODEL ################################
#############################################################



class LM_OpenAI:

    def __init__(self,model_name:str,api_key:str,file_path:str = None):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.file_path = file_path

    def generate(self,prompts, max_tokens=150,temperature=1):

        outputs=[]

        for prompt in prompts:
        
            completion = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=max_tokens,
                messages=[
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': 'Antwort:'}
                ],
                temperature=temperature
            )
            outputs.append(completion.choices[0].message.content)

        return outputs
    
    def batch_generate(self,prompts,max_tokens=150,temperature=1):

        requests = []
        for i,prompt in enumerate(prompts):

            requests.append(
                {
                    "custom_id": str(i), 
                    "method": "POST", 
                    "url": "/v1/chat/completions", 
                    "body": {"model": self.model_name,
                        "messages": [
                            {'role': 'user', 'content': prompt},
                            {'role': 'assistant', 'content': 'Antwort:'}
                        ],
                        "max_tokens": max_tokens,
                        "temperature":temperature
                    }
                }
            )

        
        file_name = self.file_path+'batches.jsonl'

        with open(file_name, 'w') as file:
            for obj in requests:
                file.write(json.dumps(obj) + '\n')

        batch_file = self.client.files.create(
            file=open(file_name, 'rb'),
            purpose='batch'
            )

        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint='/v1/chat/completions',
            completion_window='24h'
            )  
        os.remove(file_name)      

        return batch_job.id


#############################################################
######## ADD CLASS HERE WHEN USING OTHER LMs ################
#############################################################

"""
class LM_OTHER:

    def __init__(self):

    def generate(self):
        
        return outputs

