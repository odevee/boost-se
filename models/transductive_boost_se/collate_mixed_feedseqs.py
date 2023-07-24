import pandas as pd

DIR = '/cluster/project/krause/ovavourakis/boost-rs-repl/working/mixed_feedseqs1'
FILE = '/mixed_feedseqs_no_aux_yesSEP_yesMLP_d256_NSR25'

dfs = []
for block in range(10):
    for run in range(5):
        try:
            df = pd.read_csv(DIR+FILE+f'_run_{run}.csv')
            dfs.append(df)
        except:
            pass

concatenated_df = pd.concat(dfs)
averaged_df = concatenated_df.groupby(concatenated_df.index).mean(numeric_only=True)
averaged_df = averaged_df.set_index(pd.Index(['TEST']))
print(averaged_df)
print('\n')