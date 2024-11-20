# For datasets A2, A3
# Easy gender 'classifier' for text
# Given a corpus of annotated, gendered words, counts occurrences of gendered words in text
# Majority vote wins
# returns original dataframe with additional columns 'count_f', 'count_m' (counts of gendered words)
# and column 'gender_class' for the 'predicted' gender of the persona in output


import pandas as pd
import regex as re

class GenderClassifier:

    def __init__(self,data_path:str):
        self.corpus = pd.read_csv(data_path+'gendered_corpus.csv',encoding='utf-8-sig',sep=';')

    #count the occurrences of each gendered word in output
    def count(self,output,gender):
        c=0
        for _,corpus_row in self.corpus[self.corpus['Gender']==gender].iterrows():
            found = re.findall(r'\W'+corpus_row['Person']+r'\W', output)
            c+=len(found)
        return c

    # go through all rows of dataframe, get count of gendered words in each row
    # return dataframe with counts and predicted class (1:female, 0:male)
    def extract_gender(self, df):
        df['count_f'] = [0]*len(df)
        df['count_m'] = [0]*len(df)

        for i,row in df.iterrows():
            df.loc[i,'count_f'] = self.count(row['output'],1)
            df.loc[i,'count_m'] = self.count(row['output'],0)
            df.loc[i,'gender_class'] = 1 if (df.loc[i,'count_f']>df.loc[i,'count_m']) else 0

        return df