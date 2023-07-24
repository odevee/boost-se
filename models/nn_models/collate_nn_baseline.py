import pandas as pd

DIR = '/cluster/project/krause/ovavourakis/boost-rs-repl/working/trial_baseline_output'

# HORIZONTAL BASELINES
dfs = []
for i in range(10):
    df = pd.read_csv(DIR+f'/ec_nn_base_block_{i}.csv')
    dfs.append(df)

concatenated_df = pd.concat(dfs)
averaged_df = concatenated_df.groupby(concatenated_df.index).mean(numeric_only=True)
averaged_df = averaged_df.set_index(pd.Index(['EC_NN', 'ONES', 'UNIF']))
print(averaged_df)
print('\n')

dfs = []
for i in range(10):
    df = pd.read_csv(DIR+f'/embed_mean_nn_base_block_{i}.csv')
    dfs.append(df)

concatenated_df = pd.concat(dfs)
averaged_df = concatenated_df.groupby(concatenated_df.index).mean(numeric_only=True)
averaged_df = averaged_df.set_index(pd.Index(['MEMB_NN', 'ONES', 'UNIF']))
print(averaged_df)
print('\n')

dfs = []
for i in range(10):
    df = pd.read_csv(DIR+f'/embed_first_nn_base_block_{i}.csv')
    dfs.append(df)

concatenated_df = pd.concat(dfs)
averaged_df = concatenated_df.groupby(concatenated_df.index).mean(numeric_only=True)
averaged_df = averaged_df.set_index(pd.Index(['FEMB_NN', 'ONES', 'UNIF']))
print(averaged_df)
print('\n')

# VERTICAL BASELINES
dfs = []
for i in range(10):
    df = pd.read_csv(DIR+f'/FP_nn_base_block_{i}.csv')
    dfs.append(df)

concatenated_df = pd.concat(dfs)
averaged_df = concatenated_df.groupby(concatenated_df.index).mean(numeric_only=True)
averaged_df = averaged_df.set_index(pd.Index(['FP_NN', 'ONES', 'UNIF']))
print(averaged_df)
print('\n')

# DIAGONAL BASELINES
dfs = []
for i in range(10):
    df = pd.read_csv(DIR+f'/FP+EC_nn_base_block_{i}.csv')
    dfs.append(df)

concatenated_df = pd.concat(dfs)
averaged_df = concatenated_df.groupby(concatenated_df.index).mean(numeric_only=True)
averaged_df = averaged_df.set_index(pd.Index(['FP_EC_NN', 'ONES', 'UNIF']))
print(averaged_df)
print('\n')

dfs = []
for i in range(10):
    df = pd.read_csv(DIR+f'/FP+mean_nn_base_block_{i}.csv')
    dfs.append(df)

concatenated_df = pd.concat(dfs)
averaged_df = concatenated_df.groupby(concatenated_df.index).mean(numeric_only=True)
averaged_df = averaged_df.set_index(pd.Index(['FP_M_NN', 'ONES', 'UNIF']))
print(averaged_df)
print('\n')

dfs = []
for i in range(10):
    df = pd.read_csv(DIR+f'/FP+first_nn_base_block_{i}.csv')
    dfs.append(df)

concatenated_df = pd.concat(dfs)
averaged_df = concatenated_df.groupby(concatenated_df.index).mean(numeric_only=True)
averaged_df = averaged_df.set_index(pd.Index(['FP_F_NN', 'ONES', 'UNIF']))
print(averaged_df)
print('\n')