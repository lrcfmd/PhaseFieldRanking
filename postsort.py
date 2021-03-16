import pandas as pd 

df = pd.read_csv('quaternary_VAE_test_scores.csv')

#df = df.sort_values(by=['scaled'])

df =df.rename(columns={'Phase fields':'phases'})

df['lphases'] = df['phases'].apply(lambda x: x.split())
hal = ['Br'] #,'Cl', 'I', 'F']
new = []
for i, row in df.iterrows():
    for el in hal:
        if el in row['lphases'] and 'F' in row['lphases']:
            new.append(row['phases'])
            continue

df = df[df['phases'].isin(new)]
#df = df.loc[[new], ['phases', 'scores', 'scaled']]
df = df[['phases', 'scores', 'scaled']]

df =df.rename(columns={'phases':'Phase fields'})
df.to_csv('LiMFBr_testing.csv',index=False)
