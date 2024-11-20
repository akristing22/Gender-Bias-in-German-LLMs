#for dataset B2, see if 'yes' or 'no' is present in output and set column 'agreement_class' accordingly

class StatementClassifier:
    def __init__(self):
        pass

    def extract_agreement(self, dataset):
        for i,row in dataset.iterrows():
            if 'Nein' in row['output']:
                dataset.loc[i,'agreement_class'] = 0
            elif 'Ja' in row['output']:
                dataset.loc[i,'agreement_class'] = 1

        return dataset