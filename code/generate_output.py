# given the datasets and models, generate the output from the prompts

import pandas as pd
import random
from lm import LM, LM_Claude
import time
import os
import json
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# get the configuration for the analysis
with open('settings.json', 'r') as file:
    data = json.load(file)
models = data['models']
datasets = data['datasets']
output_path = data['output_path']
data_path = data['data_path']
#login token for huggingface
login_token=data['login_token_huggingface']
#api key for anthropic
anthropic_api_key=data['anthropic_api_key']
local_path=data['model_path']

#for trial runs, e.g. for prompt engineering
#don't use for final generation
def sample(data):
    rand_idx = random.sample(range(0,data.shape[0]),100)
    return data.iloc[rand_idx]


# for a model and datasets, generate all outputs and return them in 
# a new column 'output' in original dataframe
def get_output(model, datasets):

    for dataset in datasets.keys():
        #set different max_tokens for the datasets
        if dataset in ['A1','A2','A3']:
            max_t = 200
        elif dataset in ['B1','B1_control']:
            max_t = 50
        elif dataset == 'B2':
            max_t = 5

        data = datasets[dataset]
        for _,row in data.iterrows():
            data.loc[_,'output']=model.generate(row['full_prompt'],temperature=0.7,max_tokens=max_t)
        datasets[dataset] = data
    return datasets

# read all datasets defined in the settings.json file
def get_datasets():
    datasets = {}
    for dataset in data['datasets']:
        if dataset != 'A3':
            datasets[dataset] = sample(pd.read_csv(data_path+dataset+'.csv',sep=';',encoding='utf-8-sig'))
        else:
            A3 = pd.read_csv(data_path+dataset+'.csv',sep=';',encoding='utf-8-sig')
            datasets[dataset] = sample(pd.concat([A3]*20, ignore_index=True))
    return datasets


def main():
    datasets=get_datasets()

    #load each model and generate the output
    for model in models.keys():
        print(model)
        start1 = time.time()
        if model == 'Claude':
            myLM = LM_Claude(models[model],anthropic_api_key)
        else: 
            myLM = LM(local_path,models[model],login_token)
        start2 = time.time()
        datasets = get_output(myLM,datasets)

        # save the new datasets with the generated output
        for dataset in datasets.keys():
            datasets[dataset].to_csv(output_path+model+'/'+dataset+'_output.csv',sep=';',encoding='utf-8-sig',index=False)

        end = time.time()
        print('loading time: '+str(start2-start1))
        print('inference time: '+str(end-start2))

main()