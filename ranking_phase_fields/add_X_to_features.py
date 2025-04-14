import os
import pandas as pd

def addX(f):
    df = pd.read_csv(f)
    dic = {c: 0 for c in df.columns}
    xf = pd.DataFrame(dic, index=[0])
    xf['element'] = 'X'
    return pd.concat([xf,df])

for p, d, fi in os.walk('elemental_features'):
    for f in fi:
        df = addX(f'elemental_features/{f}')
        df.to_csv(f'elemental_features/{f}', index=False)
