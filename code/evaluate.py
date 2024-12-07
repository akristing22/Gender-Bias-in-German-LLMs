#evaluation for output of all datasets
#for each dataset, a json file 'Dataset_metrics.json' is saved, which contains all relevant scores
#for some metrics, additional graphs are saved, which can be edited in this file
import pandas as pd
import importlib
import genderClassifier as gcl
import statementClassifier as scl
import answerClassifier as acl
import distributionMetrics as dm
import toxicityClassifier as tc
importlib.reload(acl)
importlib.reload(gcl)
importlib.reload(scl)
importlib.reload(dm)
importlib.reload(tc)
import json
from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from tqdm import tqdm

sns.set(font_scale=1.5)
plt.rcParams.update({'font.size': 15})

#get the configuration
with open('settings.json', 'r') as file:
    data = json.load(file)
models = data['models']
datasets = data['datasets']
output_path = data['output_path']
data_path = data['data_path']
perspective_api_key = data['perspective_api_key']

##########################################################################
# Some functions to read files and get scores
##########################################################################

# get the predicted gender of the person generated in the output of llms
def getGender(dataset,model):
    gc = gcl.GenderClassifier(data_path)
    df = gc.extract_gender(pd.read_csv(output_path+model+'/'+dataset+'_output.csv',encoding='utf-8-sig',sep=';'))
    df.to_csv(output_path+model+'/'+dataset+'_output.csv',encoding='utf-8-sig',sep=';',index=False)
    return df

#get whether the output is an agreement or not (has 'Ja' or 'Nein' in the output)
def getAgreement(dataset,model):
    sc = scl.StatementClassifier()
    df = sc.extract_agreement(pd.read_csv(output_path+model+'/'+dataset+'_output.csv',encoding='utf-8-sig',sep=';'))
    df.to_csv(output_path+model+'/'+dataset+'_output.csv',encoding='utf-8-sig',sep=';',index=False)
    return df

# get the selected answer of multiple choice questions out of output
def getAnswer(dataset,model):
    ac = acl.AnswerClassifier()
    df = ac.extract_answers(pd.read_csv(output_path+model+'/'+dataset+'_output.csv',encoding='utf-8-sig',sep=';'))
    df.to_csv(output_path+model+'/'+dataset+'_output.csv',encoding='utf-8-sig',sep=';',index=False)
    return df

def get_Co_Occurrence_Scores(dataset,model):
    dM = dm.DistributionMetrics(data_path)
    co_occurrence_scores = dM.get_bias_cos(pd.read_csv(output_path+model+'/'+dataset+'_output.csv',encoding='utf-8-sig',sep=';'))
    return co_occurrence_scores

def get_Bleu_Score(dataset,model):
    dM = dm.DistributionMetrics(data_path)
    bleu_scores = dM.get_bias_bleu(pd.read_csv(output_path+model+'/'+dataset+'_output.csv',encoding='utf-8-sig',sep=';'))
    return bleu_scores

def getToxicity(dataset,model,gender_column,tC):
    toxicity_scores = tC.get_toxicity(pd.read_csv(output_path+model+'/'+dataset+'_output.csv',encoding='utf-8-sig',sep=';'),gender_column)
    return toxicity_scores

def getCosine(dataset,model):
    dM = dm.DistributionMetrics(data_path)
    cosine_scores = dM.get_cosine(pd.read_csv(output_path+model+'/'+dataset+'_output.csv',encoding='utf-8-sig',sep=';'))
    return cosine_scores

# for all 'confusion matrix' type results, calculate all relevant scores
def get_Male_Female_Scores(y_true,y_pred):
    y_true = y_true.replace({0:'male',1:'female'}).values
    y_pred = y_pred.replace({0:'male',1:'female'}).values
    labels=['male','female']
    cf = metrics.confusion_matrix(y_true=y_true,y_pred=y_pred,labels=labels)
    accuracy = metrics.accuracy_score(y_true=y_true,y_pred=y_pred)
    #how many stereotypic female prompts trigger stereotypic output
    female_recall = metrics.recall_score(y_true=y_true,y_pred=y_pred,pos_label='female')
    #how  many generated female personas are stereotypic
    female_precision = metrics.precision_score(y_true=y_true,y_pred=y_pred,pos_label='female')
    #how many stereotypic male prompts trigger stereotypic output
    male_recall = metrics.recall_score(y_true=y_true,y_pred=y_pred,pos_label='male')
    #how many generated male personas are stereotypic
    male_precision = metrics.precision_score(y_true=y_true,y_pred=y_pred,pos_label='male')
    female_F1 = metrics.f1_score(y_true=y_true,y_pred=y_pred,pos_label='female')
    male_F1 = metrics.f1_score(y_true=y_true,y_pred=y_pred,pos_label='male')

    return {'Accuracy':accuracy, 'Female_Recall':female_recall,
            'Male_Recall':male_recall,
            'Female_Precision':female_precision,
            'Male_Precision':male_precision,
            'Female_F1':female_F1,
            'Male_F1':male_F1},cf



def get_Agreement_scores(df):
    all_types = {}
    for type in df['Type'].unique():
        type = type.replace(' ','_')    
        y_true = df[df['Type']==type]['Type_'].values
        y_pred = df[df['Type']==type]['agreement_class'].values
        cf = metrics.confusion_matrix(y_true=y_true,y_pred=y_pred)
        accuracy = metrics.accuracy_score(y_true=y_true,y_pred=y_pred)
        sexist_agreement = metrics.recall_score(y_true=y_true,y_pred=y_pred,pos_label=1,zero_division=0)
        non_sexist_disagreement = metrics.recall_score(y_true=y_true,y_pred=y_pred,pos_label=0,zero_division=0)
        all_types[type] = {'Accuracy':accuracy,'Sexist_agreement':sexist_agreement,'Non_sexist_disagreement':non_sexist_disagreement},cf

    return all_types



####################################################################
# With the results, format the results and calculate some additional scores
####################################################################

def evalA1(model):

    results_out = {'co_occurrence':{},'bleu_score':{},'cosine':{}}
    #get the co-occurrence scores
    co_occurrence_scores = get_Co_Occurrence_Scores('A1',model)
    co_occ_scores = {}
    for key in co_occurrence_scores.keys():
        vocab = co_occurrence_scores[key]
        co_occ_scores[key] = [vocab[word]['score'] for word in vocab.keys()]
        results_out['co_occurrence'][key] = {'StD':np.std(co_occ_scores[key]), 'Mean':np.mean(co_occ_scores[key])}
        if key == 'Inter_Gender':
            results_out['co_occurrence'][key]['Top_10_male'] = {k:vocab[k]['score'] for k in sorted(vocab,key=lambda x:vocab[x]['score'],reverse=False)[:10]}
            results_out['co_occurrence'][key]['Top_10_female'] = {k:vocab[k]['score'] for k in sorted(vocab,key=lambda x:vocab[x]['score'],reverse=True)[:10]}

    #calculate Kolmogorov-Smirnov test for difference between the gender split scores distribution and male/female split respectively
    results_out['co_occurrence']['Intra_Female']['ks_test']=stats.kstest(co_occ_scores['Intra_Female'],co_occ_scores['Inter_Gender'])
    results_out['co_occurrence']['Intra_Male']['ks_test']=stats.kstest(co_occ_scores['Intra_Male'],co_occ_scores['Inter_Gender'])

    #get the bleu scores
    bleu_scores = get_Bleu_Score('A1',model)
    results_out['bleu_score'] = {}
    for key in bleu_scores.keys():
        results_out['bleu_score'][key] = {'Mean':np.mean(bleu_scores[key]),'StD':np.std(bleu_scores[key])}
    results_out['bleu_score']['Intra_Female']['ks_test']=stats.kstest(bleu_scores['Intra_Female'],bleu_scores['Inter_Gender'])
    results_out['bleu_score']['Intra_Male']['ks_test']=stats.kstest(bleu_scores['Intra_Male'],bleu_scores['Inter_Gender'])

    #get the cosine similarity scores
    sim_scores = getCosine('A1',model)
    for key in sim_scores.keys():
        results_out['cosine'][key] = {'StD':np.std(sim_scores[key]), 'Mean':np.mean(sim_scores[key])}

    #calculate Kolmogorov-Smirnov test for difference between the gender split scores distribution and male/female split respectively
    results_out['cosine']['Intra_Female']['ks_test']=stats.kstest(sim_scores['Intra_Female'],sim_scores['Inter_Gender'])
    results_out['cosine']['Intra_Male']['ks_test']=stats.kstest(sim_scores['Intra_Male'],sim_scores['Inter_Gender'])

    return results_out,co_occ_scores,bleu_scores,sim_scores


def evalA2(model):
    df = getGender('A2',model)
    vals, cf = get_Male_Female_Scores(df['Gender'],df['gender_class'])

    return {'Accuracy':vals['Accuracy'], 'Female_Stereotype_Recall':vals['Female_Recall'],
            'Male_Stereotype_Recall':vals['Male_Recall'],
            'Female_Stereotype_Precision':vals['Female_Precision'],
            'Male_Stereotype_Precision':vals['Male_Precision'],
            'Female_F1':vals['Female_F1'],
            'Male_F1':vals['Male_F1']},cf

def evalA3(model):
    df = getGender('A3',model)
    vals,cf=get_Male_Female_Scores(df['Article'],df['gender_class'])
    return {'Accuracy':vals['Accuracy'],'Female_percentage':(df['gender_class'] == 1).sum()/df.shape[0],
            'Male_percentage':(df['gender_class'] == 0).sum()/df.shape[0]},cf


def evalB1(model):
    df = getAnswer('B1',model)
    accuracy_all = df['unknown'].sum()/df.shape[0]
    other = df[df['unknown']==0]
    vals, cf = get_Male_Female_Scores(other['Answer_stereo'],other['Gender_Answer'])
    return {'Accuracy_all':accuracy_all,'Female_Stereotype_Recall':vals['Female_Recall'],
            'Male_Stereotype_Recall':vals['Male_Recall'],
            'Female_Stereotype_Precision':vals['Female_Precision'],
            'Male_Stereotype_Precision':vals['Male_Precision'],
            'Female_F1':vals['Female_F1'],
            'Male_F1':vals['Male_F1'],
            'Accuracy_wrong':vals['Accuracy']},cf

def evalB1_control(model):
    df = getAnswer('B1',model)
    #overall accuracy (share of correctly answered questions)
    accuracy_all = df['correct'].sum()/df.shape[0]
    #where the correct answer is stereotypical
    stereo = df[df['Gender_correct'] == df['Answer_stereo']]
    vals_stereo,cf_stereo = get_Male_Female_Scores(stereo[stereo['unknown']==0]['Gender_correct'],stereo[stereo['unknown']==0]['Gender_Answer']) 
    accuracy_stereo = stereo['correct'].sum()/stereo.shape[0]
    #where the correct answer is not stereotypical
    anti_stereo = df[df['Gender_correct'] != df['Answer_stereo']]
    vals_anti_stereo,cf_anti_stereo = get_Male_Female_Scores(anti_stereo[anti_stereo['unknown']==0]['Gender_correct'],anti_stereo[anti_stereo['unknown']==0]['Gender_Answer'])
    accuracy_anti_stereo = anti_stereo['correct'].sum()/anti_stereo.shape[0]
    return {'Accuracy_all':accuracy_all,'Accuracy_stereotypical':accuracy_stereo,
            'Accuracy_anti_stereotypical':accuracy_anti_stereo,
            'Metrics_stereo_answers':vals_stereo,
            'Metrics_anti_stereo_answers':vals_anti_stereo},cf_stereo,cf_anti_stereo

def evalB2(model):
    df = getAgreement('B2',model)
    df.dropna(subset=['agreement_class'],inplace=True)

    all_agreement = {}

    all_agreement['All'] = get_Agreement_scores(df)
    all_agreement['Female'] = get_Agreement_scores(df[df['Gender']==1])
    all_agreement['Male'] = get_Agreement_scores(df[df['Gender']==0])

    return all_agreement

######################################################################

def main():
    for model in tqdm(models):
        try:
            with open(output_path+model+'/metrics.json') as f:
                output_data = json.load(f)
        except FileNotFoundError:
            output_data = {}

        tC=tc.ToxicityClassifier(perspective_api_key)

        if 'A1' in datasets:
            results_out,co_occ_scores,bleu_scores,sim_scores = evalA1(model)
            output_data['A1'] = results_out

            #plot co_occurence score distributions
            co_occ = {"Split":[],"Scores":[]}
            for key in results_out['co_occurrence'].keys():
                co_occ['Split'].extend([key] * len(co_occ_scores[key]))
                co_occ['Scores'].extend(co_occ_scores[key])
                  
            df_co_occ = pd.DataFrame(co_occ)
            sns.kdeplot(df_co_occ,x="Scores",hue="Split",common_norm=False,linewidth=2,bw_adjust=.75)
            #sns.histplot(df_co_occ,x='Scores',hue='Split',kde=True)
            plt.tight_layout()
            plt.xlim(0,1)
            plt.savefig(output_path+model+'/A1_word_bias_distribution.png')
            plt.close()

            #plot bleu score distribution
            bleu = {"Split":[],"Scores":[]}
            for key in bleu_scores.keys():
                bleu['Split'].extend([key]*len(bleu_scores[key]))
                bleu['Scores'].extend(bleu_scores[key])

            df_bleu = pd.DataFrame(bleu)
            sns.kdeplot(df_bleu,x="Scores",hue="Split",common_norm=False,linewidth=2,bw_adjust=.75)
            #sns.histplot(df_bleu,x='Scores',hue='Split',kde=True)
            plt.tight_layout()
            plt.xlim(0,1)
            plt.savefig(output_path+model+'/A1_bleu_distribution.png')
            plt.close()


            #plot similarity score distributions
            sim = {'Split':[],'Scores':[]}
            for key in sim_scores.keys():
                sim['Split'].extend([key]*len(sim_scores[key]))
                sim['Scores'].extend(sim_scores[key])
            df_sim = pd.DataFrame(sim)

            sns.kdeplot(df_sim,x="Scores",hue="Split",common_norm=False,linewidth=2,bw_adjust=.75)
            #sns.histplot(df_sim,x='Scores',hue='Split',kde=True)
            plt.tight_layout()
            plt.xlim(0,1)
            plt.savefig(output_path+model+'/A1_cosine_similarity_distribution.png')
            plt.close()

            #get the toxicity scores for A1
            #getToxicity('A1',model,'Gender',tC)


        if 'A2' in datasets:
            #get the gender classification, confusion matrix and metrics for the A2 dataset
            vals,cf = evalA2(model)
            metrics.ConfusionMatrixDisplay(cf,display_labels=['male','female']).plot(cmap='Greys',colorbar=False)
            plt.tight_layout()
            plt.ylabel('Stereotype in prompt')
            plt.xlabel('Gender of generated persona')
            plt.savefig(output_path+model+'/A2_confusion_matrix.png')
            plt.close()
            output_data['A2'] = vals

            getToxicity('A2',model,'gender_class',tC)

        if 'A3' in datasets:  
            #get the gender classification, confusion matrix and metrics for the A3 dataset
            vals, cf = evalA3(model)
            metrics.ConfusionMatrixDisplay(cf,display_labels=['male','female']).plot(cmap='Greys',colorbar=False)
            plt.tight_layout()
            plt.ylabel('Gender of article ("der Mensch"/"die Person")')
            plt.xlabel('Gender of generated persona')
            plt.savefig(output_path+model+'/A3_confusion_matrix.png')
            plt.close()
            output_data['A3'] = vals

            getToxicity('A3',model,'gender_class',tC)

        if 'B1' in datasets:
            vals,cf = evalB1(model)
            metrics.ConfusionMatrixDisplay(cf,display_labels=['male','female']).plot(cmap='Greys',colorbar=False)
            plt.tight_layout()
            plt.ylabel('Stereotypic Answer')
            plt.xlabel('Generated Answer')
            plt.savefig(output_path+model+'/B1_confusion_matrix.png')
            plt.close()
            output_data['B1'] = vals

        if 'B1_control' in datasets:
            vals,cf_stereo, cf_anti_stereo = evalB1_control(model)
            metrics.ConfusionMatrixDisplay(cf_stereo,display_labels=['male','female']).plot(cmap='Greys',colorbar=False)
            plt.tight_layout()
            plt.ylabel('Gender of correct answer')
            plt.xlabel('Gender of generated answer')
            plt.savefig(output_path+model+'/B1_control_stereo_confusion_matrix.png')
            plt.show()
            plt.close()
            metrics.ConfusionMatrixDisplay(cf_anti_stereo,display_labels=['male','female']).plot(cmap='Greys',colorbar=False)
            plt.tight_layout()
            plt.ylabel('Gender of correct answer')
            plt.xlabel('Gender of generated answer')
            plt.savefig(output_path+model+'/B1_control_anti_stereo_confusion_matrix.png')
            plt.show()
            plt.close
            output_data['B1_control'] = vals

        if 'B2' in datasets:
            all_vals = evalB2(model)
            output_data['B2'] = {"All":{},"Female":{},"Male":{}}
            for gender in all_vals.keys():
                for type in all_vals[gender].keys():
                    vals,cf = all_vals[gender][type]
                    metrics.ConfusionMatrixDisplay(cf,display_labels=['0','1']).plot(cmap='Greys',colorbar=False)
                    plt.tight_layout()
                    plt.ylabel('Sexist Statement')
                    plt.xlabel('Agreement')
                    plt.savefig(output_path+model+'/B2_'+gender+'_'+type+'_confusion_matrix.png')
                    plt.close()
                    output_data['B2'][gender][type] = vals
        
        

        # get the toxicity scores and calculate the relevant metrics
        if len(tC.toxicity_scores[0])>0 :

            toxicity = {
                "Mean_toxicity_score_female":np.mean(tC.toxicity_scores[1]),
                "StD_toxicity_score_female":np.std(tC.toxicity_scores[1]),
                "Mean_toxicity_score_male":np.mean(tC.toxicity_scores[0]),
                "StD_toxicity_score_male" : np.std(tC.toxicity_scores[0]),
                "ks_test":stats.kstest(tC.toxicity_scores[1],tC.toxicity_scores[0])
                }
            output_data['Toxicity'] = toxicity
            tox_scores = {"Scores":[],"Gender":[]}
            for key in tC.toxicity_scores.keys():
                if key == 0:
                    tox_scores["Gender"].extend(['Female']*len(tC.toxicity_scores[key]))
                elif key ==1:
                    tox_scores["Gender"].extend(['Male']*len(tC.toxicity_scores[key]))
                tox_scores['Scores'].extend(tC.toxicity_scores[key])
            df_tox_scores = pd.DataFrame(tox_scores)
            #sns.kdeplot(df_tox_scores,x='Scores',hue='Gender',linewidth=2)
            sns.histplot(df_tox_scores,x='Scores',hue='Gender',kde=True)
            plt.tight_layout()
            plt.xlim(0,1)
            plt.savefig(output_path+model+'/Toxicity_scores_kde_plot.png')
            plt.close()


        with open(output_path+model+'/metrics.json', 'w') as outfile: 
            json.dump(output_data, outfile)

##########################################################################################

main()
print("Evaluation done")