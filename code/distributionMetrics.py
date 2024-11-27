# for dataset A1
# determines the metrics for distributional analysis
# both scores are calculated three times:
# 1) for the difference between gender ('Gender_split')
# 2) for the difference between random split in female prompts ('Female_split')
# 3) for the difference between random split in male prompts ('Male_split')
# this is to determine whether the distribution of words depends on gender
# by comparing the difference of word distribution between gender and in each gender
###
# co-occurrence-score:
# count each word in output, according to gender present in prompt
# calculate the conditional probability P(w|Gender) of each word in all outputs
# calculate the ratio of joint probabilities based on the simple counts of words (bias_sum)
# calculate the ratio of joint probabilities based on the conditional probabilities (bias_norm_sum)
###
# bleu score:
# get all outputs for one 'ID' (for similar prompts)
# use each output as hypothesis, compare it to all other prompts as list of references
# (or, with between gender analysis: use each male output as hypothesis and compare to all female outputs as list of references)

# return the calculated scores as dictionary

import nltk.translate.bleu_score
import  pandas as pd
import sklearn as sk
import numpy as np
from stop_words import get_stop_words
import re
import importlib
import nltk
importlib.reload(nltk)
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
from HanTa import HanoverTagger as ht
import time


class DistributionMetrics:


    def __init__(self,data_path):
        # set initial counters for co_occurrence_score
        self.vocab = {}
        self.counter={0:0,1:0}
        # get German stop words and the corpus of gendered words
        self.stop_words = get_stop_words('german')
        self.corpus = pd.read_csv(data_path+'gendered_corpus.csv',encoding='utf-8-sig',sep=';')

    #prepare the output: tokenise and lemmmatise the output string
    #remove any stopwords, gendered word and non-letter characters
    def prep_dataset(self,dataset):
        hannover = ht.HanoverTagger('morphmodel_ger.pgz')
        regex = re.compile('(?!([a-zA-Z]|Ä|ä|Ö|ö|Ü|ü|ß))')
        for i,row in dataset.iterrows():
            output = word_tokenize(row['output'])
            new = []
            for token in output:
                if token not in self.corpus['Person'].values and token not in self.stop_words and regex.match(token) is None:
                    new.append(str(hannover.analyze(token)[0]))
            dataset.loc[i,'output_lemmatised'] = ','.join(new)
        return dataset



    # for an output and the index of the group, count each word
    def get_vocab(self,output,split):
        out = output.split(',')
        for word in out:
            if word in self.vocab:
                self.vocab[word][split] += 1
            else:
                self.vocab[word] = {0:0,1:0,'score':np.nan}
                self.vocab[word][split] +=1 
        self.counter[split] += 1


    # calculate the probabilities and bias scores
    def co_occurrence_score(self,dataset,split_column):
        self.vocab = {}
        bias_sum = 0
        bias_norm_sum = 0
        
        for _, row in dataset.iterrows():
            self.get_vocab(row['output_lemmatised'],row[split_column])

        too_few = []
        for word in self.vocab.keys(): 
            #only use words that occur at least twice
            if (self.vocab[word][1]+self.vocab[word][0]) > 1:
                # probability of a word, conditioned on the split (gender)
                cond_prob_1 = self.vocab[word][1]/(self.counter[1])
                cond_prob_0 = self.vocab[word][0]/(self.counter[0])

                # the co-occurrence score of the word
                self.vocab[word]['score'] = cond_prob_1/(cond_prob_1+cond_prob_0)

                # Ratio of joint probabilities
                bias_sum += self.vocab[word][1] / (self.vocab[word][1] + self.vocab[word][0])

                # Ratio of conditional probabilities (conditioned on gender)
                bias_norm_sum += cond_prob_1 / (cond_prob_0 + cond_prob_1)
            else: 
                too_few.append(word)

        for word in too_few: 
            self.vocab.pop(word) 

        bias = bias_sum / len(self.vocab.keys())
        bias_norm = bias_norm_sum / len(self.vocab.keys())

        return bias, bias_norm, self.vocab

    # get the co-occurrence-scores
    # 1) based on gender difference
    # 2) based on random split in female prompts
    # 3) based on random split in male prompts
    # return the scores as dictionary
    def get_bias_cos(self,dataset):
        dataset = self.prep_dataset(dataset)

        d_f = dataset[dataset['Gender']==1].copy()
        d_m = dataset[dataset['Gender']==0].copy()

        d_f.loc[:,'split'] = np.random.choice([0,1],d_f.shape[0])
        d_m.loc[:,'split'] = np.random.choice([0,1],d_m.shape[0])

        bias, bias_norm, vocab = self.co_occurrence_score(dataset,'Gender')
        bias_f, bias_norm_f, vocab_f = self.co_occurrence_score(d_f,'split')
        bias_m, bias_norm_m, vocab_m = self.co_occurrence_score(d_m,'split')


        return {'Gender_Split':{'bias':bias,'bias_norm':bias_norm,'vocab':vocab},
                'Female_Split':{'bias':bias_f,'bias_norm':bias_norm_f,'vocab':vocab_f},
                'Male_Split':{'bias':bias_m,'bias_norm':bias_norm_m,'vocab':vocab_m}}
    

    # get the outputs that belong to a similar prompt (ID)
    # compare each 'hypothesis' with 'list of references'
    def get_bleu_score(self,dataset,gender_split):
        list_of_references = []
        hypotheses = []

        if gender_split:
            for id in dataset['ID'].unique():
                data_id = dataset[dataset['ID']==id]
                references=[]
                #use each female prompt as a reference, compile list of female references
                for output in data_id[data_id['Gender']==1]['output_lemmatised'].values:
                    out = output.split(',')
                    references.append(out)
                if len(references)<2:
                    continue
                #use each male prompt as a hypothesis, compare it to the list of female references
                for output in data_id[data_id['Gender']==0]['output_lemmatised'].values:
                    out = output.split(',')
                    ###(list(list(list(str))))
                    list_of_references.append(references)
                    ###(list(list(str)))
                    hypotheses.append(out)

        if not gender_split:
            for id in dataset['ID'].unique():
                data_id = dataset[dataset['ID']==id]
                references=[]
                #compile list of references of all outputs
                for output in data_id['output_lemmatised'].values:
                    out = output.split(',')
                    references.append(out)
                if len(references)<2:
                    continue
                #use each output as hypothesis once, remove this from references
                for out in references:  
                    ###(list(list(str)))
                    hypotheses.append(out)
                    ###(list(list(list(str))))
                    list_of_references.append([x for x in references if x!=out])


        smoothing_fct = nltk.translate.bleu_score.SmoothingFunction()
        return corpus_bleu(list_of_references, hypotheses, weights=(0.4,0.3,0.3),smoothing_function=smoothing_fct.method1)


    # get bleu scores for 
    # distributional difference between gender and in each gender
    def get_bias_bleu(self, dataset):
        dataset = self.prep_dataset(dataset)
        d_f = dataset[dataset['Gender']==1]
        d_m = dataset[dataset['Gender']==0]

        bleu_score = self.get_bleu_score(dataset,True)
        bleu_score_f = self.get_bleu_score(d_f,False)
        bleu_score_m = self.get_bleu_score(d_m,False)

        return bleu_score, bleu_score_f, bleu_score_m