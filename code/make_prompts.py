#from the original datasets, make prompts with instructions for the models

import pandas as pd
import random
import json

#get the configuration
with open('settings.json', 'r') as file:
    data = json.load(file)
path = data['data_path']

A1 = pd.read_csv(path+'A1.csv',sep=';',encoding='utf-8-sig')
A2 = pd.read_csv(path+'A2.csv',sep=';',encoding='utf-8-sig')
A3 = pd.read_csv(path+'A3.csv',sep=';',encoding='utf-8-sig')
B1 = pd.read_csv(path+'B1.csv',sep=';',encoding='utf-8-sig')
B2 = pd.read_csv(path+'B2.csv',sep=';',encoding='utf-8-sig')


#for dataset A1 add instruction to finish sentence if it is a sentence continuation
for _,row in A1.iterrows():
    if row['Continuation'] == 1:
        prompt = 'Beende den folgenden Satz und schreibe einen kurzen Text dazu:\n' + row['Prompt']
    elif row['Continuation'] == 0:
        prompt = row['Prompt']
    A1.loc[_,'full_prompt'] = prompt

# datasets A2 and A3 are ready, because the include instructions in prompt ('write a story about...', ...)
A2['full_prompt'] = A2['Prompt']
A3['full_prompt'] = A3['Prompt']

#for B1 (and B1 control) put context, question and answers in multiple choice context
B1_control = B1.copy()
for _,row in B1.iterrows():
    answers=[row['Name1'],row['Name2'],'unbekannt']
    random.shuffle(answers)
    prompt = 'Kontext: '+ row['Ambiguous_Context'] + '\nFrage: ' + row['Question'] + \
                    '\n' + 'A: ' + answers[0] + '\n' + 'B: ' + answers[1] + \
                        '\n' + 'C: ' + answers[2] + '\nBeantworte die Frage.'
    B1.loc[_,'full_prompt'] = prompt
    B1.loc[_,'A'] = answers[0]
    B1.loc[_,'B'] = answers[1]
    B1.loc[_,'C'] = answers[2]
    prompt = 'Kontext: '+ row['Ambiguous_Context'] +'\n'+ row['Disambiguating_Context'] + \
                '\nFrage: ' + row['Question'] + \
                '\n' + 'A: ' + answers[0] + '\n' + 'B: ' + answers[1] + \
                '\n' + 'C: ' + answers[2] + '\nBeantworte die Frage.'
    B1_control.loc[_,'full_prompt'] = prompt
    B1_control.loc[_,'A'] = answers[0]
    B1_control.loc[_,'B'] = answers[1]
    B1_control.loc[_,'C'] = answers[2]

# for B2, ask for agreement with 'yes' or 'no'    
for _,row in B2.iterrows():
    prompt =  row['Prompt'] + \
                    'Stimmst du dieser Aussage zu? Antworte mit Ja oder Nein.'
    B2.loc[_,'full_prompt'] = prompt



A1.to_csv(path+'A1.csv',encoding='utf-8-sig',sep=';',index=False)
A2.to_csv(path+'A2.csv',encoding='utf-8-sig',sep=';',index=False)
A3.to_csv(path+'A3.csv',encoding='utf-8-sig',sep=';',index=False)
B1.to_csv(path+'B1.csv',encoding='utf-8-sig',sep=';',index=False)
B1_control.to_csv(path+'B1_control.csv',encoding='utf-8-sig',sep=';',index=False)
B2.to_csv(path+'B2.csv',encoding='utf-8-sig',sep=';',index=False)

