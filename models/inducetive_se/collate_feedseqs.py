import pandas as pd

DIR = '/cluster/project/krause/ovavourakis/boost-rs-repl/working/feedseqs8'
FILE = '/all_aux_no_autoenc_yes_MLP_yes_SEP_d2048_meanembeds_NSR1'

dfs = []
for block in range(10):
    for run in range(5):
        try:
            df = pd.read_csv(DIR+FILE+f'_block_{block}_run_{run}.csv')
            dfs.append(df)
        except:
            pass

concatenated_df = pd.concat(dfs)
averaged_df = concatenated_df.groupby(concatenated_df.index).mean(numeric_only=True)
averaged_df = averaged_df.set_index(pd.Index(['DTEST', 'HTEST', 'VTEST', 'HVTEST', 'VVTEST']))
print(averaged_df)
print('\n')