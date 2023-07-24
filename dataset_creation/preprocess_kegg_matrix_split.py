import pandas as pd
import numpy as np
import torch
import pickle
from rdkit.Chem import MACCSkeys, MolFromMolFile
from myutil import encode_KOs, encode_EC

import matplotlib.pyplot as plt
from tqdm import tqdm


np.random.seed(3) # to decide which negative training examples to place in validation and test sets

KEGG        = '/Users/odysseasvavourakis/Documents/2022-2023/Studium/5. Semester/Thesis Work/datasets.nosync/kegg/'
EC_KO_FILE  = KEGG+'ligand/enzyme/links/enzyme_ko.list'
MOL_FILES   = KEGG+'ligand/compound/mol'
ES_FILE     = KEGG+'ligand/compound/links/compound_enzyme.list'
CC_FILE     = KEGG+'ligand/rclass/rclass_cpair.lst'

# ignore these simple/common compounds
cp_blacklist = {'C00001'} # water

trainset_include_cpds = {'C00002','C00003','C00004','C00005','C00006','C00007','C00008','C00009','C00010',
                'C00011','C00012','C00013','C00014','C00015','C00016','C00017','C00018','C00019','C00020',
                'C00021','C00022','C00023','C00024','C00027','C00029','C00032','C00034','C00035','C00038',
                'C00044'}

print('GETTING E/S INTERACTIONS')
esi = pd.read_csv(ES_FILE, sep='\t', header=None)
esi[0]=esi[0].str.split(':').apply(lambda list : list[1])  # drop cpd: prefix
esi[1]=esi[1].str.split(':').apply(lambda list : list[1])  # drop ec: prefix
esi = esi.set_axis(['compound', 'EC'], axis=1)

print('GETTING SUBSTRATE FINGERPRINTS')
substrates = set(esi['compound'])
mol_FPs = {}
for S in substrates:
    path = MOL_FILES+'/'+S+'.mol'
    try:
        rep = MolFromMolFile(MOL_FILES+'/'+S+'.mol')
        mol_FPs[S] = MACCSkeys.GenMACCSKeys(rep)
    except: # generic functions, photons, macromolecular, faulty files
        cp_blacklist.add(S)

print('GETTING COMPOUND-COMPOUND REACTION RELATIONS')
cc = pd.read_csv(CC_FILE, sep='\t', header=None).drop(axis=1,columns=[0,2])
cc = cc[1].str.split('_').apply(lambda list: pd.Series([list[0],list[1]]))
# also consider reverse pairs
cc_rev = pd.concat([cc[1],cc[0]], axis=1).set_axis([0,1], axis=1)
cc = pd.concat([cc,cc_rev], axis=0, ignore_index=True)
# create compound : compound interaction dictionary
CC = cc.groupby(by=cc[0]).agg(set).to_dict()[1]

print('GETTING ENZYME KOs')
ec_ko = pd.read_csv(EC_KO_FILE, sep='\t', header=None)
ec_ko[0]=ec_ko[0].str.split(':').apply(lambda list : list[1]) # drop ec: prefix
ec_ko[1]=ec_ko[1].str.split(':').apply(lambda list : list[1]) # drop ko: prefix
ec_ko = ec_ko.groupby(by=ec_ko[0]).agg(set)

print('FILTERING DOWN ENZYMES & COMPOUNDS')
# consider enzymes with BOTH an interaction AND an assigned KO; drop others
enzymes = set(esi['EC']) & set(ec_ko.index)
ec_ko = ec_ko.drop(set(ec_ko.index)-enzymes, axis=0)
esi = esi.drop(esi[esi['EC'].apply(lambda ec: not ec in enzymes)].index)
# consider substrates with ALL of (fingerprints, CC relations, ESI); drop others
substrates = (substrates - cp_blacklist) & set(CC.keys()) & set(esi['compound'])
for key in set(mol_FPs.keys())-substrates:
    mol_FPs.pop(key)
for key in set(CC.keys())-substrates:
    CC.pop(key)
esi = esi[esi['compound'].apply(lambda cmp: cmp in substrates)]
# last step has removed more enzymes
enzymes = set(esi['EC']) & set(ec_ko.index)
ec_ko = ec_ko.drop(set(ec_ko.index)-enzymes, axis=0)
esi = esi.drop(esi[esi['EC'].apply(lambda ec: not ec in enzymes)].index)
assert(set(esi['compound']) == substrates)
assert(set(mol_FPs.keys()) == substrates)
assert(set(CC.keys()) == substrates)
assert(enzymes == set(esi['EC']))
assert(enzymes == set(ec_ko.index))

# analyse EC numbers, set up encode/decoder dictionaries
As, Bs, Cs = set(),set(),set()
for E in enzymes:
    A, B, C, _ = [int(i) for i in E.split('.')]
    As.add(A); Bs.add(B); Cs.add(C)
len_ECa, len_ECb, len_ECc = [len(x) for x in [As, Bs, Cs]]
ECa_to_hot, ECb_to_hot, ECc_to_hot = \
        [ { EC_field : one_hot for EC_field, one_hot in zip(list(field), list(np.identity(len(field), dtype=int))) } for field in [As, Bs, Cs]]
hot_to_ECa, hot_to_ECb, hot_to_ECc = \
        [ { tuple(v): k for k, v in dic.items()} for dic in [ECa_to_hot, ECb_to_hot, ECc_to_hot] ]

# create dictionary EC : {compounds}
enzymes, substrates, num_compound, num_enzyme = sorted(enzymes), sorted(substrates), len(substrates), len(enzymes)
E_S = esi.groupby(by=esi['EC']).agg(set).to_dict()['compound']

print('CREATING INTERACTION MATRIX')
matrix = np.zeros((num_compound,num_enzyme), dtype=bool)

Eid_to_EC = {} # maps numerical enzyme id to EC
EC_to_Eid = {} # maps ECs back to numerical enzyme ids

Sid_to_cmpd = {} # maps numerial compound id to KEGG C_id
cmpd_to_Sid = {} # maps KEGG C_id to numerical compound id

for E in E_S:
    for S in E_S[E]:
        assert(S in substrates)
        E_index, S_index = enzymes.index(E), substrates.index(S)

        Eid_to_EC[E_index], EC_to_Eid[E] = E, E_index

        Sid_to_cmpd[S_index], cmpd_to_Sid[S] = S, int(S_index)

        matrix[S_index,E_index] = 1

# plt.figure(dpi=2000)
# plt.imshow(matrix, cmap='binary', vmin=0, vmax=1, interpolation='none')
# plt.xlabel('enzymes')
# plt.ylabel('compounds')
# plt.title('Interaction Data')
# plt.savefig('all_kegg_data.png', dpi=1000, bbox_inches='tight')

print('BUILDING SUBSTRATE FINGERPRINT TABLE')
fps = pd.DataFrame.from_dict(mol_FPs, orient='index').reset_index().set_axis(['compound', 'fingerprint'], axis=1)
fps['fingerprint'] = fps['fingerprint'].apply(lambda fp: np.array(fp))
fps['compound'] = fps['compound'].apply(lambda cp: cmpd_to_Sid[cp])
fps = fps.sort_values('compound').set_index('compound')

print('ENCODING EC TABLE')
ko_set = set().union(*[set for set in ec_ko[1]])
num_ko = len(ko_set)
EC_table = pd.DataFrame.from_dict(Eid_to_EC, orient='index')
EC_table=EC_table[0].apply(lambda EC_num: encode_EC(EC_num, (ECa_to_hot, ECb_to_hot, ECc_to_hot)))

print('BUILDING CC PAIR TABLE AND DICTIONARY')
pairlist = []
for anchor, paired_compounds in CC.items():
    for P in paired_compounds:
        if anchor in substrates and P in substrates:
            pairlist.append([cmpd_to_Sid[anchor],cmpd_to_Sid[P]])
cc_pairs = np.array(pairlist)
CC_with_Sids = {cmpd_to_Sid[key]: {cmpd_to_Sid[value] for value in values if value in substrates} for key, values in CC.items() if key in substrates}

print('ENCODING KOs')
# build pos : KO dictionary
pos_to_ko, ko_to_pos = {}, {}
for i, ko in enumerate(sorted(ko_set)):
    pos_to_ko[i] = ko
    ko_to_pos[ko] = i
# encode each enzyme's KO vector (multi-hot)
ec_ko = ec_ko.reset_index()
ec_ko[0] = ec_ko[0].apply(lambda ec : EC_to_Eid[ec])
ec_ko[1] = ec_ko[1].apply(lambda kos : encode_KOs(kos, ko_to_pos, num_ko))
ec_ko.set_index(0)




# randomly subsample rows and columns for training, validation, testing
print('PERFORMING DATA SPLITS and BUILDING INTERACTION TABLES')
for seed in range(10):
    np.random.seed(seed)
    print('---- training set')
    # include common cofactors in the trainset by default
    trainset_include_Sids = np.array([cmpd_to_Sid[cpd] for cpd in trainset_include_cpds if cpd in cmpd_to_Sid.keys()])
    num_cpds_to_select = int(0.84 * num_compound)
    num_enzs_to_select = int(0.84 * num_enzyme)     # this way we train on 0.84^2 = 70% of the cells

    tr_enz_idx = np.random.choice(matrix.shape[1], size=num_enzs_to_select, replace=False)

    tr_cpd_idx = np.random.choice(matrix.shape[0], size=num_cpds_to_select, replace=False)
    mask = np.isin(tr_cpd_idx, trainset_include_Sids) # avoid duplicates in trainset compound ids
    tr_cpd_idx = np.concatenate((trainset_include_Sids, tr_cpd_idx[~mask]))

    tr = pd.DataFrame(matrix).iloc[tr_cpd_idx,tr_enz_idx]   # subset matrix
    tr = tr.where(tr, other=False).stack().reset_index()    # get which indices are 1, which 0 (as "list")

    print('---- validation sets')
    remaining_cpds = np.setdiff1d(np.arange(matrix.shape[0]), tr_cpd_idx)
    remaining_enzs = np.setdiff1d(np.arange(matrix.shape[1]), tr_enz_idx)
    num_cpds_to_select = int(0.66 * len(remaining_cpds))
    num_enzs_to_select = int(0.66 * len(remaining_enzs))
    va_cpd_idx = np.random.choice(remaining_cpds, size=num_cpds_to_select, replace=False)
    va_enz_idx = np.random.choice(remaining_enzs, size=num_enzs_to_select, replace=False)

    d_va = pd.DataFrame(matrix).iloc[va_cpd_idx, va_enz_idx]        # subset matrix
    d_va = d_va.where(d_va, other=False).stack().reset_index()      # get which indices are 1, which 0 (as "list")

    h_va = pd.DataFrame(matrix).iloc[tr_cpd_idx, va_enz_idx]        # subset matrix
    h_va = h_va.where(h_va, other=False).stack().reset_index()      # get which indices are 1, which 0 (as "list")

    v_va = pd.DataFrame(matrix).iloc[va_cpd_idx, tr_enz_idx]        # subset matrix
    v_va = v_va.where(v_va, other=False).stack().reset_index()      # get which indices are 1, which 0 (as "list")

    print('---- test sets')
    remaining_cpds = np.setdiff1d(remaining_cpds, va_cpd_idx)
    remaining_enzs = np.setdiff1d(remaining_enzs, va_enz_idx)

    d_te = pd.DataFrame(matrix).iloc[remaining_cpds, remaining_enzs]        # subset matrix
    d_te = d_te.where(d_te, other=False).stack().reset_index()              # get which indices are 1, which 0 (as "list")

    h_te = pd.DataFrame(matrix).iloc[tr_cpd_idx, remaining_enzs]            # subset matrix
    h_te = h_te.where(h_te, other=False).stack().reset_index()              # get which indices are 1, which 0 (as "list")

    v_te = pd.DataFrame(matrix).iloc[remaining_cpds, tr_enz_idx]            # subset matrix
    v_te = v_te.where(v_te, other=False).stack().reset_index()              # get which indices are 1, which 0 (as "list")

    h_va_te = pd.DataFrame(matrix).iloc[va_cpd_idx, remaining_enzs]         # subset matrix
    h_va_te = h_va_te.where(h_va_te, other=False).stack().reset_index()     # get which indices are 1, which 0 (as "list")

    v_va_te = pd.DataFrame(matrix).iloc[remaining_cpds, va_enz_idx]         # subset matrix
    v_va_te = v_va_te.where(v_va_te, other=False).stack().reset_index()     # get which indices are 1, which 0 (as "list")

    for s in [tr, d_va, h_va, v_va, d_te, h_te, v_te, h_va_te, v_va_te]:
        s.columns = ['compound', 'enzyme', 'interaction']
        s['interaction'] = s['interaction'].astype(int)
        s = s.sort_values('interaction', ascending=False, inplace=True, ignore_index=True)
    tr, d_va, h_va, v_va, d_te, h_te, v_te, h_va_te, v_va_te = [tr, d_va, h_va, v_va, d_te, h_te, v_te, h_va_te, v_va_te]

    # convert everything to torch tensors
    tr = torch.from_numpy(tr.to_numpy()).to(dtype=torch.int32)
    d_va = torch.from_numpy(d_va.to_numpy()).to(dtype=torch.int32)
    h_va = torch.from_numpy(h_va.to_numpy()).to(dtype=torch.int32)
    v_va = torch.from_numpy(v_va.to_numpy()).to(dtype=torch.int32)
    d_te = torch.from_numpy(d_te.to_numpy()).to(dtype=torch.int32)
    h_te = torch.from_numpy(h_te.to_numpy()).to(dtype=torch.int32)
    v_te = torch.from_numpy(v_te.to_numpy()).to(dtype=torch.int32)
    h_va_te = torch.from_numpy(h_va_te.to_numpy()).to(dtype=torch.int32)
    v_va_te = torch.from_numpy(v_va_te.to_numpy()).to(dtype=torch.int32)

    fps_out = torch.from_numpy(np.asarray(list(fps['fingerprint']))).to(dtype=torch.int32)
    EC_table_out = torch.from_numpy(np.asarray(list(EC_table))).to(dtype=torch.int32)
    cc_pairs_out = torch.from_numpy(cc_pairs).to(dtype=torch.int32)
    ec_ko_out = torch.from_numpy(np.asarray(list(ec_ko[1]))).to(dtype=torch.int32)

    interaction_data = { 'tr' : tr,
                         'd_va' : d_va,
                         'h_va' : h_va,
                         'v_va' : v_va,
                         'd_te' : d_te,
                         'h_te' : h_te,
                         'v_te' : v_te,
                         'h_va_te' : h_va_te,
                         'v_va_te' : v_va_te,

                         'num_compound' : num_compound,
                         'num_enzyme' : num_enzyme,
                         'fp_label' : fps_out,
                         'ec_label' : EC_table_out,
                         'len_EC_fields' : (len_ECa, len_ECb, len_ECc),
                         'EC_to_hot_dicts' : [ECa_to_hot, ECb_to_hot, ECc_to_hot],
                         'hot_to_EC_dicts' : [hot_to_ECa, hot_to_ECb, hot_to_ECc],
                         'pos_to_ko_dict' : pos_to_ko,
                         'ko_to_pos_dict' : ko_to_pos,
                         'num_ko' : num_ko,
                         'rpairs_pos' : cc_pairs_out,
                         'CC_dict' : CC_with_Sids,
                         'enzyme_ko_hot' : ec_ko_out
    }

    # dump to file
    with open(f'./data/block_splits/interaction_real_cofacs-train_block_{seed}.pkl', 'wb') as fi:
        pickle.dump(interaction_data, fi, protocol=pickle.HIGHEST_PROTOCOL)

with open('./data/block_splits/enzyme_key.pkl', 'wb') as fi:
    pickle.dump(Eid_to_EC, fi, protocol=pickle.HIGHEST_PROTOCOL)

with open('./data/block_splits/substrate_key.pkl', 'wb') as fi:
    pickle.dump(Sid_to_cmpd, fi, protocol=pickle.HIGHEST_PROTOCOL)