import torch
import pickle
import numpy as np
import sklearn.metrics as sk_m
from torch.utils.data import Dataset

import time

def encode_EC(EC_NUMBER, conversion_dictionaries):
    ECa_to_hot, ECb_to_hot, ECc_to_hot = conversion_dictionaries
    A, B, C, _ = [int(i) for i in EC_NUMBER.split('.')]
    veca, vecb, vecc = [field[value] for field, value in zip([ECa_to_hot, ECb_to_hot, ECc_to_hot], [A, B, C])]
    return np.concatenate([veca, vecb, vecc])


def decode_EC(EC_hot_vec, conversion_dictionaries, lentuple):
    len_ECa, len_ECb, len_ECc = lentuple
    hot_to_ECa, hot_to_ECb, hot_to_ECc = conversion_dictionaries
    veca, vecb, vecc = [EC_hot_vec[:len_ECa], EC_hot_vec[len_ECa:len_ECa + len_ECb], EC_hot_vec[len_ECa + len_ECb:]]
    integers = [field[tuple(value)] for field, value in zip([hot_to_ECa, hot_to_ECb, hot_to_ECc], [veca, vecb, vecc])]
    return ".".join([str(i) for i in integers]) + ".-"


def encode_KOs(kos, ko_to_pos, num_ko):
    # multi-hot encoding
    enc = np.zeros(num_ko)
    for ko in kos:
        enc[ko_to_pos[ko]] = 1
    return enc

def decode_KOs(vector, pos_to_ko):
    return [pos_to_ko[index[0]] for index in np.where(vector == 1)]

def load_interaction_data(fpath):
    with open(fpath, 'rb') as fi:
        data = pickle.load(fi)

        pairs = data['pairs']
        # pos_pairs = data['pos_pairs']
        # neg_pairs = data['neg_pairs']
        num_compound = data['num_compound']
        num_enzyme = data['num_enzyme']
        fp_label = data['fp_label']
        ec_label = data['ec_label']
        len_EC_fields = data['len_EC_fields']
        EC_to_hot_dicts = data['EC_to_hot_dicts']
        hot_to_EC_dicts = data['hot_to_EC_dicts']
        pos_to_ko_dict = data['pos_to_ko_dict']
        ko_to_pos_dict = data['ko_to_pos_dict']
        num_ko = data['num_ko']
        CC_dict = data['CC_dict']
        enzyme_ko_hot = data['enzyme_ko_hot']

    return pairs, num_compound, num_enzyme, \
        fp_label, ec_label, len_EC_fields, EC_to_hot_dicts, hot_to_EC_dicts, \
        pos_to_ko_dict, ko_to_pos_dict, num_ko, CC_dict, enzyme_ko_hot
    # return pos_pairs, neg_pairs, num_compound, num_enzyme, \
    #         fp_label, ec_label, len_EC_fields, EC_to_hot_dicts, hot_to_EC_dicts, \
    #         pos_to_ko_dict, ko_to_pos_dict, num_ko, CC_dict, enzyme_ko_hot


def load_block_interaction_data(fpath):
    with open(fpath, 'rb') as fi:
        data = pickle.load(fi)

        tr = data['tr']
        d_va = data['d_va']
        h_va = data['h_va']
        v_va = data['v_va']
        d_te = data['d_te']
        h_te = data['h_te']
        v_te = data['v_te']
        h_va_te = data['h_va_te']
        v_va_te = data['v_va_te']

        num_compound = data['num_compound']
        num_enzyme = data['num_enzyme']
        fp_label = data['fp_label']
        ec_label = data['ec_label']
        len_EC_fields = data['len_EC_fields']
        EC_to_hot_dicts = data['EC_to_hot_dicts']
        hot_to_EC_dicts = data['hot_to_EC_dicts']
        pos_to_ko_dict = data['pos_to_ko_dict']
        ko_to_pos_dict = data['ko_to_pos_dict']
        num_ko = data['num_ko']
        CC_dict = data['CC_dict']
        enzyme_ko_hot = data['enzyme_ko_hot']

    return tr, d_va, h_va, v_va, d_te, h_te, v_te, h_va_te, v_va_te, num_compound, num_enzyme, \
        fp_label, ec_label, len_EC_fields, EC_to_hot_dicts, hot_to_EC_dicts, \
        pos_to_ko_dict, ko_to_pos_dict, num_ko, CC_dict, enzyme_ko_hot

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data  # compound enzyme {0,1}
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        es_interaction = self.data[index].unsqueeze(0)
        compound_id = es_interaction[:, 0]
        enzyme_id = es_interaction[:, 1]
        return es_interaction, compound_id, enzyme_id

class CustomDataset2(Dataset):
    # apart from interactions, also samples CC (anchor, pos, neg) triplets
    def __init__(self, posdata, negdata, negrate=1, sample_triplets=False, CC=None, num_cpd=None):
        self.posdata = posdata  # compound enzyme 1
        self.negdata = negdata  # compound enzyme 0
        self.negrate = negrate  # negative sampling ratio for main task
        self.CC = CC  # anchor : {positive1, positive2} (have cc relation)
        self.num_cpd = num_cpd  # total number of compounds in dataset
        self.sample_triplets = sample_triplets

    def __len__(self):
        return self.posdata.shape[0]

    def __getitem__(self, index):
        # sample pos and neg es-ixns 1:negrate
        es_pos_ixn = self.posdata[index].unsqueeze(0)
        es_neg_ixn = self.negdata[torch.randint(0,self.negdata.shape[0],(self.negrate,)), :]
        if len(es_neg_ixn.shape) == 1: es_neg_ixn = es_neg_ixn.unsqueeze(0)
        es_ixns = torch.cat([es_pos_ixn,es_neg_ixn], dim=0)
        cpd_ids = es_ixns[:, 0]
        enz_ids = es_ixns[:, 1]
        if self.sample_triplets:
            # get all (anchor, positive) cc relations for compounds involved (auxilliary task)
            # sample ONE negative per anchor (this is unrelated to the neg_rate which controls the main task)
            # note: the original paper sampled 25 negs, instead of one
            triplets = []
            for cpd in cpd_ids:
                positives = torch.Tensor(list(self.CC[int(cpd)]))
                anchors = torch.Tensor(np.repeat(int(cpd), len(positives)))
                negatives = torch.from_numpy(np.random.choice(np.arange(self.num_cpd), len(positives)))
                trplts = torch.stack([anchors, positives, negatives], dim=1).long()
                triplets.append(trplts)
            triplets = torch.cat(triplets, dim=0).long()
            return es_ixns, cpd_ids, enz_ids, triplets
        return es_ixns, cpd_ids, enz_ids

def custom_collate(batch):
    objs, compound_ids, enzyme_ids, triplets = [], [], [], []
    for obj, comp_ids, enz_ids, *triplet in batch:
        objs.append(obj)
        compound_ids.append(comp_ids)
        enzyme_ids.append(enz_ids)
        triplets.append(triplet[0]) if triplet else None
    objs = torch.cat(objs, dim=0)
    compound_ids = torch.cat(compound_ids, dim=0)
    enzyme_ids = torch.cat(enzyme_ids, dim=0)
    if triplets:
        return objs, compound_ids, enzyme_ids, torch.cat(triplets, dim=0)
    else:
        return objs, compound_ids, enzyme_ids

def tree_distance(ec1, ec2):
    ec1_parts, ec2_parts = [ec.strip().split('.') for ec in [ec1, ec2]]
    distance = 0
    for level in range(3):
        if ec1_parts[level] != ec2_parts[level]:
            distance += 10**(2-level)  # weights: (100, 10, 1, None) in EC: A.B.C.D

    distance += np.abs(int(ec1_parts[-1])-int(ec2_parts[-1]))

    return distance

def cos_sim(tensor1, tensor2):
    dot_product = torch.dot(tensor1, tensor2)
    norm_tensor1 = torch.norm(tensor1)
    norm_tensor2 = torch.norm(tensor2)
    similarity = dot_product / (norm_tensor1 * norm_tensor2)
    return similarity.item()

def inv_euclidean_distance(tensor1, tensor2):
    squared_diff = torch.pow(tensor1 - tensor2, 2)
    sum_squared_diff = torch.sum(squared_diff)
    distance = torch.sqrt(sum_squared_diff)
    return 1/distance

def report_metric(num_compound, num_enzyme, true_interaction, pred_interaction, te_pn):
    metric = {}

    # compute map
    def map(n, dim, k=None):
        # precision for particular compound, averaged over all compounds or mutatis mutandis enzymes
        rst = []
        for i in range(n):
            indices = te_pn[:, dim] == i
            if indices.sum() == 0: continue
            x = true_interaction[indices]
            y = pred_interaction[indices]
            if k is not None:
                y_sorted_indices = np.argsort(-y)[:k]
                x = x[y_sorted_indices]
                y = y[y_sorted_indices]
                if x.sum() == 0:
                    rst.append(0)
                    continue
            rst.append(sk_m.average_precision_score(y_true=x, y_score=y))
        rst = (np.mean(rst), np.std(rst) / np.sqrt(len(rst)))
        return rst

    metric['compound_map'] = map(num_compound, 0, k=None)
    metric['enzyme_map'] = map(num_enzyme, 1, k=None)

    metric['compound_map_3'] = map(num_compound, 0, k=3)
    metric['enzyme_map_3'] = map(num_enzyme, 1, k=3)

    # compute r precision and precision@k(1, 3)
    def precision(n, k=None, dim=None):
        def h(x, y):
            m_true = int(x.sum()) if k is None else k
            if m_true == 0: return -1

            xy = np.vstack([x, y]).T
            xy_sorted_indices = np.argsort(-xy[:, 1])
            xy = xy[xy_sorted_indices]

            z = xy[:m_true, 0].sum() / m_true

            return z

        rst = []
        if dim is None:
            x = true_interaction
            y = pred_interaction
            z = h(x, y)
            if z != -1:
                rst.append(z)
        else:
            for i in range(n):
                indices = te_pn[:, dim] == i
                if indices.sum() == 0: continue
                x = true_interaction[indices]
                y = pred_interaction[indices]
                z = h(x, y)
                if z != -1:
                    rst.append(z)
        if k is None and dim is None:
            rst = np.mean(rst)
        else:
            rst = (np.mean(rst), np.std(rst) / np.sqrt(len(rst)))
        return rst

    metric['compound_rprecision'] = precision(num_compound, k=None, dim=0)
    metric['enzyme_rprecision'] = precision(num_enzyme, k=None, dim=1)
    metric['rprecision'] = precision(num_enzyme, k=None, dim=None)

    metric['compound_precision_1'] = precision(num_compound, k=1, dim=0)
    metric['enzyme_precision_1'] = precision(num_enzyme, k=1, dim=1)

    return metric
