#for dataset A1,2,3
#get the perspective API toxicity score for each output
#returns dictionary with list of scores for keys 0 and 1 (male and female)
from googleapiclient.errors import HttpError
from googleapiclient import discovery
import time


class ToxicityClassifier:

    #initialise the perspecitve client
    def __init__(self, api_key):
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
            )
        self.toxicity_scores = {0:[],1:[]}

    # query perspective api for score
    def get_score(self,output):
        analyze_request = {
            'comment': { 'text': output },
            'requestedAttributes': {'TOXICITY': {}},
            'languages':['de']
            }
        try:
            response = self.client.comments().analyze(body=analyze_request).execute()
            score = response['attributeScores']['TOXICITY']['summaryScore']['value']
        except HttpError as E:
            print(E)
            print(type(E))
            print()
            try:
                if E.error_details[0]['reason'] == "RATE_LIMIT_EXCEEDED":
                    time.sleep(0.1)
                    score = self.get_score(output)
                else:
                    score = None
            except KeyError as ER:
                print(output)
                score = None
        
        return score
    
    #for all outputs, get scores
    def get_toxicity(self,dataset,gender_column):

        for _,row in dataset.iterrows():
            score = self.get_score(row['output'])  
            if not score is None:        
                self.toxicity_scores[row[gender_column]].append(score)
