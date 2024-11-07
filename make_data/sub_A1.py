import pandas as pd

#male=0, female=1

#read all relevant files (templates, and substitutions)
A1_templates = pd.read_csv("../data/A1_templates.csv",encoding="utf-8-sig",sep=";")
Names = pd.read_csv("../data/Names.csv",encoding="utf-8-sig",sep=";")
Occupations = pd.read_csv("../data/Occupations.csv",encoding="utf-8-sig",sep=";")
General = pd.read_csv("../data/General.csv",encoding="utf-8-sig",sep=";")

#generated prompt and gender
gender=[]
prompts=[]
cont=[]
subs=[]


def create(row,p,source):
    prompt = row["Template"].replace("[P]",p)
    pr = prompt.split()
    pr[0]=pr[0].capitalize()
    prompt = " ".join(pr)

    gender.append(row["Gender"])
    prompts.append(prompt)
    cont.append(row["Continuation"])
    subs.append(source)


for _,rowG in General.iterrows():
    for _,row in A1_templates[(A1_templates["Gender"]==rowG["Gender"]) & (A1_templates["Case"]==rowG["Case"])].iterrows():
            create(row,rowG["Person"],"General")

for _,rowO in Occupations.iterrows():
    for _,row in A1_templates[(A1_templates["Gender"]==rowO["Gender"]) &
                               (A1_templates["Case"]==rowO["Case"]) & 
                               (A1_templates["Occupation"]==1)].iterrows():
            create(row,rowO["Person"],"Occupation")

for _,rowN in Names.iterrows():
    for _,row in A1_templates[(A1_templates["Gender"]==rowN["Gender"])].iterrows():
            create(row,rowN["Person"],"Name")

A1 = pd.DataFrame({"Prompt":prompts,"Gender":gender,"Continuation":cont,"Person":subs})
A1.to_csv("../data/A1.csv",index=False,encoding="utf-8-sig",sep=";")
