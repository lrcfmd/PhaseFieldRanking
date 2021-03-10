import pandas as pd 

df = pd.read_csv('quinary_AE_testing_scores_nice.csv')

df = df.sort_values(by=['scaled'])

df =df.rename(columns={'Phase fields':'phases'})

df['lphases'] = df['phases'].apply(lambda x: x.split())

new = []
for i, row in df.iterrows():
    if 'O' in row['lphases'] and 'S' in row['lphases']:
       new.append(row['phases'])

df = df[df['phases'].isin(new)]
#df = df.loc[[new], ['phases', 'scores', 'scaled']]
df = df[['phases', 'scores', 'scaled']]

df =df.rename(columns={'phases':'Phase fields'})
df.to_csv('quinary_OS.csv',index=False)
