# In this file, a model from huggingface/Anthropic can be loaded and used.
# Only works if huggingface model is supported by transformer architecture and AutoTokenizer and AutoModelForCausalLM 
# Depending on the model used, a new class has to be written so it can be used in the "generate_prompts.py" code
# class has to have a generate() function, which takes the prompt, max_tokens and temperature as as variables

from huggingface_hub import snapshot_download
from huggingface_hub import login
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import anthropic
import torch
import time
from openai import OpenAI

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
   
    def download(self):
        login(token=self.login_token)
        snapshot_download(repo_id=self.model_name, local_dir=self.dir)

    # load model from local file
    def load(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.dir,low_cpu_mem_usage=True,device_map='auto',torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2')
        self.tokenizer = AutoTokenizer.from_pretrained(self.dir,low_cpu_mem_usage=True,device_map='auto')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        
        
    def generate(self, prompt:str, max_tokens=150, temperature=1):
        message = prompt + '\nAntwort:\n'
        inputs = self.tokenizer(message,return_tensors='pt',return_attention_mask=True).to('cuda')
        attention_mask = inputs['attention_mask']
        generate_ids = self.model.generate(inputs.input_ids,temperature=temperature, attention_mask=attention_mask,max_new_tokens=max_tokens,do_sample=True)
        output = self.tokenizer.decode(generate_ids[0],skip_special_tokens=True)
        return output.split('Antwort:')[-1]

# lm class for a claude model from anthropic
class LM_Claude:

    def __init__(self, model_name:str, api_key:str):
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt:str, max_tokens=150,temperature=1):
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

        return output
    

class LM_OpenAI:

    def __init__(self,model_name:str,api_key:str):
        self.model_name = model_name
        self.client = OpenAI()

    def generate(self,prompt:str, max_tokens=150,temperature=1):
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': 'Antwort:'}
            ]
        )
        output = completion.choices[0].message
        return output




