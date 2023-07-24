"""
BLOCK DATA NN BASELINE

BASH LAUNCHER:

for i in {0..9}; do
        sbatch -A ls_krausea -n 1 --cpus-per-task=1 --gpus=1 --gres=gpumem:16384m --time=4:00:00 --mem-per-cpu=16384 --wrap="python nneigh_baseline.py --DATA $i"
done
"""

import torch, copy, argparse, os, pickle
import pandas as pd
import numpy as np

from tqdm import tqdm

from myutil import load_block_interaction_data, tree_distance, cos_sim, inv_euclidean_distance
from sklearn.metrics import confusion_matrix

# def train(model):
#
#     print(( f"{'TYPE':<5s} {'EPOCH':<5s} {'NAM':5s} {'T-LOSS':>10s} {'V-LOSS':>10s} {'AUROC':>10s} "
#             f"{'AP':>10s} {'MAX':>10s} {'MIN':>10s} {'#(+)(.5)':>10s} {'%(+)(.5)':>11s} {'fREC':>10s} "
#             f"{'H#(+)':>10s} {'H100':>10s} {'H30':>10s} {'E100':>10s} {'E30':>10s}"
#     ))
#
#     for epoch in range(args.EPOCHS):
#         model.train()
#
#         # # manual batched implementation -> TODO: adjust to match the non-batched one (after fusion event)
#         # batch_size = 670000
#         # for bi in range(int(np.ceil(train_data.shape[0] / batch_size))):
#         #     indices_s = bi * batch_size
#         #     indices_e = min(train_data.shape[0], (bi + 1) * batch_size)
#         #     cpds = train_data[indices_s:indices_e, 0].long()
#         #     enzs = train_data[indices_s:indices_e, 1].long()
#         #     # get data
#         #     obs_ixns = train_data[indices_s:indices_e, -1].long()
#         #     cpd_set = torch.unique(cpds)
#         #     enz_set = torch.unique(enzs)
#         #     # forward
#         #     pred_ixns = model(cpds, enzs)
#         #     # loss
#         #     loss = multi_task_loss(model, pred_ixns, obs_ixns, cpd_set, enz_set, epoch)
#         #     # backprop
#         #     opt.zero_grad()
#         #     loss.backward()
#         #     opt.step()
#         #     # eval
#         #     if epoch % args.EVAL_FREQ == 0 and bi == 0:
#         #         # mtl, auroc, auprc = evaluate(model, train_data[:int(0.4 * train_data.shape[0]), :], loss, epoch=epoch)
#         #         mtl, auroc, auprc = evaluate(model, val_data, loss, epoch=epoch, bi=bi)
#         #         if auprc > best_auprc:
#         #             best_auprc = auprc
#         #             epoch_best_auprc = epoch
#         #             best_model_state = copy.deepcopy(model.state_dict())
#
#         # non-batched implementation
#         # get interaction data
#         num_pos, num_neg = one_data.shape[0], zero_data.shape[0]
#         neg_indices = torch.randperm(num_neg)[:args.NSR * num_pos]
#         random_negatives = zero_data[neg_indices]
#         train_batch = torch.cat([one_data, random_negatives], dim=0)
#         obs_ixns, cpds, enzs = train_batch[:, -1].float(), train_batch[:, 0].long(), train_batch[:, 1].long()  # indices must be int64, hence long()
#         # forward
#         pred_ixns = model(cpds, enzs)
#         # loss
#         loss = multi_task_loss(model, pred_ixns, obs_ixns, epoch)
#         # backprop
#         opt.zero_grad()
#         loss.backward()
#         # print(f'GRADIENTS:\t {[p.grad for p in model.parameters() if p.requires_grad]}\n')
#         opt.step()
#         # eval
#         if epoch % args.EVAL_FREQ == 0:
#             enrich_factors = []
#             for data, name in [(hval_data, 'HVAL'), (vval_data, 'VVAL'), (dval_data, 'DVAL')]:
#                 enrichments = evaluate(model, data, loss, epoch=epoch, type=name)
#                 enrich_factors.append(enrichments)
#             enrich_factors = torch.tensor(enrich_factors).float()
#             mean_enrichments = torch.mean(enrich_factors, dim=0)
#             median_enrichtments = torch.median(enrich_factors, dim=0)[0]
#
#             # inverse_sparsities_tensor = torch.tensor([[inverse_sparsities[2],       # hval
#             #                                            inverse_sparsities[3],       # vval
#             #                                            inverse_sparsities[1]]])     # dval
#             # hits = torch.tensor(hit_numbers).float() * inverse_sparsities_tensor
#             # avg_hits = torch.tensor(hit_numbers).float().sum(0)/sum(inverse_sparsities_tensor)
#
#             print(f'AVERAGE ENRICHMENTS: {mean_enrichments}\n')
#             print(f'MEDIAN ENRICHMENTS: {median_enrichtments}\n')
#             if mean_enrichments[0] > most_mean_enrichment:  # comparing enrichment_100
#                 most_mean_enrichment, epoch_most_mean_enrichment, best_model_state = mean_enrichments[0], epoch, copy.deepcopy(model.state_dict())
#
#         # # freeze embedding params after a while
#         # if epoch == args.T:
#         #     print(f'\n FREEZING EMB WEIGHTS FROM EPOCH {epoch_best_auprc}')
#         #     model.load_state_dict(best_model_state)
#         #     if model.sep_embeds:
#         #         components_to_freeze = [model.MF_Embedding_Compound.parameters(),
#         #                                 model.MF_Embedding_Enzyme.parameters(),
#         #                                 model.MLP_Embedding_Compound.parameters(),
#         #                                 model.MLP_Embedding_Enzyme.parameters()]
#         #     else:
#         #         components_to_freeze = [model.Embedding_Compound.parameters(),
#         #                                 model.Embedding_Enzyme.parameters()]
#         #     for component in components_to_freeze:
#         #         for param in component:
#         #             param.requires_grad = False
#
#     print(f'\nFINISHED TRAINING -- TESTING BEST MODEL (from epoch {epoch_most_avg_hits})')
#     model.load_state_dict(best_model_state)
#     datas = [dte_data, hte_data, vte_data, h_va_te_data, v_va_te_data]
#     names = ['DTEST', 'HTEST', 'VTEST', 'HVTEST', 'VVEST']
#     for data, name in zip(datas, names):
#         evaluate(model, data, loss, epoch=-1, type=name)


def evaluate(test_data, true_ixns, pred_ixns, outfile):
    # naive baseline predictions
    all_ones = torch.ones_like(true_ixns)
    unif_rand = torch.rand_like(all_ones.float())  # uniformly random on [0,1]
    # convert prediction vectors to numpy
    ypreds = [y.numpy() for y in [pred_ixns, all_ones, unif_rand]]
    eval_list = zip(['MODEL', ' ONES', ' UNIF'], ypreds)

    true_ixns = true_ixns.numpy()

    print((f"{'TYPE':<5s} {'NAM':5s} {'PPV':>10s} "
           f"{'fREC':>10s} {'E#(+)':>10s} {'E100':>10s} {'H100':>10s}"
    ))

    results = {}
    for name, ypred in eval_list:
        # threshold-specific metrics
        ypred_bin_5 = np.where(ypred > 0.5, 1.0, 0.0)  # binarise predictions with threshold = 0.5
        tn, fp, fn, tp = confusion_matrix(true_ixns, ypred_bin_5).ravel()

        ppv = tp / (tp + fp)   # positive predictive value (= precision); lower bound wrt unknown ground truth
        frec = tp / (tp + fn)  # fraction of observed (+) recovered; no specific relationship to ground-truth TPR

        # enrichment score
        num_obs_pos = int(true_ixns.sum())              # number of observed (+) interactions
        posrate = num_obs_pos / true_ixns.shape[0]      # chance to get a hit at random (in 1 draw)

        sorted_pred_indices = np.argsort(ypred)[::-1]                               # predictions DEscending by logits
        indices = [sorted_pred_indices[:i] for i in [num_obs_pos, 100]]             # top #observed(+), 100 preds
        yield_npos, yield_100 = [int(true_ixns[index].sum()) for index in indices]  # hits in those predictions

        enrich_npos = yield_npos / (num_obs_pos * posrate)  # enrichment relative to random guessing
        enrich_100 = yield_100 / (100 * posrate)

        # output
        type = 'HORI'

        print((f"{type:5s} {name:5s} {ppv:>10.3f} "
               f"{frec:>10.4f} {enrich_npos:>10.1f} {enrich_100:>10.1f} {yield_100:>10d} "
        ))

        resultvals = (type, name, ppv, frec, enrich_npos, enrich_100, yield_100)
        results[name] = resultvals

        # save correct hit pairs
        def save_hits(top_pred_indices):
            top_pred_data = test_data[top_pred_indices.copy()]  # copy-workaround for np limitation (no neg strides!)
            hit_rows = top_pred_data[top_pred_data[:, -1] == 1]

            out = pd.DataFrame(hit_rows, columns=['substrate', 'enzyme', 'observed interaction'])

            out['substrate'] = out['substrate'].apply(lambda s_id: substrate_key[s_id])
            out['enzyme'] = out['enzyme'].apply(lambda e_id: enzyme_key[e_id])

            out.to_csv("".join([cwd, '/', args.OUT, '/', outfile, f'_{args.DATA}_hits{len(top_pred_indices)}.txt']), index=False)

        if name == 'MODEL':
            save_hits(indices[1])  # correct hits from top 100 predictions

    # save model eval stats on the test set (for this block)
    colnames = ['TYPE', 'NAME', 'PPV', 'fRec', 'E#(+)', 'E100', 'H100']
    df = pd.DataFrame(columns=colnames)

    for type in results.keys():
        row = results[type]
        df.loc[len(df)] = row

    df.to_csv("".join([cwd, '/', args.OUT, '/', outfile, f'_{args.DATA}.csv']), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="(EC)-Nearest-Neighbour Baseline for Real System")

    # system and files
    parser.add_argument('--DEVICE', type=str, default='cuda')
    parser.add_argument('--DATA', type=int, default=0)                      # which block split dataset
    parser.add_argument('--OUT', type=str, default='trial_baseline_output')

    # # training parameters
    # parser.add_argument('--PI', type=float, default=0.01)                   # assumed true fraction of (+) ixns
    # parser.add_argument('--RHO', type=float, default=0.1)                   # of these, fraction that are observed
    # parser.add_argument('--ALPHA', type=float, default=1)                   # assym. penalty for FN
    # parser.add_argument('--LOSS', choices=['BCE', 'PROB1'], default='BCE')  # main task loss

    args = parser.parse_args()

    print('\n---- SYSTEM PARAMETERS ----')
    print(f'GPU:\t\t\t {torch.cuda.is_available()}')
    print(f'DATA:\t\t\t block_{args.DATA}')

    # print('\n---- TRAINING PARAMETERS ----')
    # print(f'PI:\t\t\t {args.PI}')
    # print(f'RHO:\t\t\t {args.RHO}')
    # print(f'ALPHA:\t\t\t {args.ALPHA}')
    # print(f'MAIN LOSS:\t\t\t {args.LOSS}')

    # global definitions
    # system
    cwd = os.getcwd()                       # working directory
    os.makedirs(args.OUT, exist_ok=True)    # output directory

    # enzyme and substrate entities: {internal_id : EC or KEGG compound}
    with open('/cluster/home/ovavourakis/projects/src/data/block_splits/substrate_key.pkl', 'rb') as fi:
        substrate_key = pickle.load(fi)
    with open('/cluster/home/ovavourakis/projects/src/data/block_splits/enzyme_key.pkl', 'rb') as fi:
        enzyme_key = pickle.load(fi)

    # interaction data
    file = f'/cluster/home/ovavourakis/projects/src/data/block_splits/interaction_real_cofacs-train_block_{args.DATA}.pkl'
    tr, d_va, h_va, v_va, d_te, h_te, v_te, h_va_te, v_va_te, \
        N_CPD, N_ENZ, fp_label, ec_label, len_EC_fields, \
        EC_to_hot_dicts, hot_to_EC_dicts, pos_to_ko_dict, ko_to_pos_dict, \
        num_ko, CC_dict, enzyme_ko_hot = load_block_interaction_data(file)
    ixn_datasets = [tr, d_va, h_va, v_va, d_te, h_te, v_te, h_va_te, v_va_te]
    train_data, dval_data, hval_data, vval_data, dte_data, hte_data, vte_data, h_va_te_data, v_va_te_data = ixn_datasets

    h_data = hte_data         # horizontal test set
    v_data = vte_data         # vertical test set
    d_data = dte_data         # diagonal test set

    try:
        pickle_dir = 'pickled_nn_bls/'
        with open(pickle_dir+f'ec_nn_preds_block_{args.DATA}.pickle', 'rb') as file:
            pred_ixns_ec = pickle.load(file)

        with open(pickle_dir+f'mean_enz_embed_nn_preds_block_{args.DATA}.pickle', 'rb') as file:
            pred_ixns_embed_mean = pickle.load(file)

        with open(pickle_dir+f'first_enz_embed_nn_preds_block_{args.DATA}.pickle', 'rb') as file:
            pred_ixns_embed_first = pickle.load(file)

        with open(pickle_dir+f'FP_nn_preds_block_{args.DATA}.pickle', 'rb') as file:
            pred_ixns_FP = pickle.load(file)

        with open(pickle_dir+f'FP+EC_nn_preds_block_{args.DATA}.pickle', 'rb') as file:
            pred_ixns_FP_EC = pickle.load(file)

        with open(pickle_dir+f'FP+mean_nn_preds_block_{args.DATA}.pickle', 'rb') as file:
            pred_ixns_FP_mean = pickle.load(file)

        with open(pickle_dir+f'FP+first_nn_preds_block_{args.DATA}.pickle', 'rb') as file:
            pred_ixns_FP_first = pickle.load(file)

        print("Pickle files found. Loaded dataframes.")

    except FileNotFoundError:
        # construct training data 'matrix' from list
        print('Reconstructing Training Data Matrix (Lookup Table)')
        tr_matrix = pd.DataFrame(train_data.numpy(), columns=['sid', 'eid', 'value'])
        tr_matrix = tr_matrix.pivot(index='sid', columns='eid', values='value')

        # enz embeddings
        file1 = f'/cluster/project/krause/ovavourakis/boost-rs-repl/data/seq_embeds/esm2_3B_mean.pkl'
        with open(file1, 'rb') as f:
            raw_enz_embeds_mean = pickle.load(f).to(args.DEVICE)
        file2 = f'/cluster/project/krause/ovavourakis/boost-rs-repl/data/seq_embeds/esm2_3B_first.pkl'
        with open(file2, 'rb') as f:
            raw_enz_embeds_first = pickle.load(f).to(args.DEVICE)

        # cpd embeddings
        fp_label = fp_label.double().to(args.DEVICE)

        # calculate FP similarity matrix
        try:
            with open(f'pickled_nn_bls/sim_matrix_FP_block_{args.DATA}.pickle', 'rb') as file:
                sim_matrix_FP = pickle.load(file)

        except FileNotFoundError:
            print('Calculating FP Similarity Matrix')

            sid_trs = torch.unique(train_data[:, 0])  # rows
            sid_vs = torch.unique(v_data[:, 0])       # columns

            sim_matrix_FP = torch.empty((len(sid_trs), len(sid_vs)))

            for i, sid_tr in tqdm(enumerate(sid_trs)):
                FP_tr = fp_label[sid_tr.item()]
                for j, sid_v in enumerate(sid_vs):
                    FP_v = fp_label[sid_v.item()]

                    sim_matrix_FP[i, j] = inv_euclidean_distance(FP_tr, FP_v)
                    # sim_matrix_FP[i, j] = cos_sim(FP_tr, FP_v)

            with open(f'sim_matrix_FP_block_{args.DATA}.pickle', 'wb') as file:
                pickle.dump(sim_matrix_FP, file)

        sim_matrix_FP = pd.DataFrame(sim_matrix_FP)
        sim_matrix_FP.columns, sim_matrix_FP.index = sid_vs.tolist(), sid_trs.tolist()
        closest_sids_FP = sim_matrix_FP.idxmax(axis=0)  # closest sid_tr for each sid_v by embedding cos_sim

        # calculate enzyme distance / similarity matrices
        print('Calculating EC and enz_embed Similarity/Distance Matrix')

        eid_trs = torch.unique(train_data[:, 1])                     # rows
        eid_hs = torch.unique(h_data[:, 1] )                         # columns

        dist_matrix = torch.empty( (len(eid_trs), len(eid_hs)) )
        sim_matrix_mean = torch.empty( (len(eid_trs), len(eid_hs)) )
        sim_matrix_first = torch.empty( (len(eid_trs), len(eid_hs)) )

        for i, eid_tr in tqdm(enumerate(eid_trs)):
            EC_tr = enzyme_key[eid_tr.item()]
            mean_embed_eid_tr = raw_enz_embeds_mean[eid_tr.item()]
            first_embed_eid_tr = raw_enz_embeds_first[eid_tr.item()]
            for j, eid_h in enumerate(eid_hs):
                EC_h = enzyme_key[eid_h.item()]
                mean_embed_eid_h = raw_enz_embeds_mean[eid_h.item()]
                first_embed_eid_h = raw_enz_embeds_first[eid_h.item()]

                dist_matrix[i, j] = tree_distance(EC_tr, EC_h)
                # sim_matrix_mean[i, j] = cos_sim(mean_embed_eid_tr, mean_embed_eid_h)
                # sim_matrix_first[i, j] = cos_sim(first_embed_eid_tr, first_embed_eid_h)
                sim_matrix_mean[i, j] = inv_euclidean_distance(mean_embed_eid_tr, mean_embed_eid_h)
                sim_matrix_first[i, j] = inv_euclidean_distance(first_embed_eid_tr, first_embed_eid_h)

        dist_matrix = pd.DataFrame(dist_matrix)
        dist_matrix.columns, dist_matrix.index = eid_hs.tolist(), eid_trs.tolist()
        closest_eids = dist_matrix.idxmin(axis=0)                    # closest eid_tr for each eid_h by EC tree-dist

        sim_matrix_mean = pd.DataFrame(sim_matrix_mean)
        sim_matrix_mean.columns, sim_matrix_mean.index = eid_hs.tolist(), eid_trs.tolist()
        closest_eids_embed_mean = sim_matrix_mean.idxmax(axis=0)     # closest eid_tr for each eid_h by embedding cos_sim

        sim_matrix_first = pd.DataFrame(sim_matrix_first)
        sim_matrix_first.columns, sim_matrix_first.index = eid_hs.tolist(), eid_trs.tolist()
        closest_eids_embed_first = sim_matrix_first.idxmax(axis=0)   # closest eid_tr for each eid_h by embedding cos_sim

        print('predicting targets using closest enzyme in train-set')

        def lookup_at_nn(row, type = 'EC'):
            sid, eid, true = row

            if type in ['FP', 'FP+EC', 'FP+mean', 'FP+first']:
                closest_sid = closest_sids_FP[sid]
                closest_eid = eid
                if type == 'FP+EC':
                    closest_eid = closest_eids[eid]
                elif type == 'FP+mean':
                    closest_eid = closest_eids_embed_mean[eid]
                elif type == 'FP+first':
                    closest_eid = closest_eids_embed_first[eid]

            elif type in ['EC', 'embed_mean', 'embed_first']:
                closest_sid = sid
                if type == 'EC':
                    closest_eid = closest_eids[eid]
                elif type == 'embed_mean':
                    closest_eid = closest_eids_embed_mean[eid]
                elif type == 'embed_first':
                    closest_eid = closest_eids_embed_first[eid]

            return tr_matrix.loc[closest_sid, closest_eid]

        # interaction predictions by ec_nn-lookup
        pred_ixns_ec = pd.DataFrame(h_data).apply(lambda row: lookup_at_nn(row), axis=1)
        pred_ixns_ec = torch.tensor(pred_ixns_ec.values)

        with open(pickle_dir+f'ec_nn_preds_block_{args.DATA}.pickle', 'wb') as file:
            pickle.dump(pred_ixns_ec, file)

        # interaction predictions by enz_embed_mean_nn-lookup
        pred_ixns_embed_mean = pd.DataFrame(h_data).apply(lambda row: lookup_at_nn(row, type='embed_mean'), axis=1)
        pred_ixns_embed_mean = torch.tensor(pred_ixns_embed_mean.values)

        with open(pickle_dir+f'mean_enz_embed_nn_preds_block_{args.DATA}.pickle', 'wb') as file:
            pickle.dump(pred_ixns_embed_mean, file)

        # interaction predictions by enz_embed_first_nn-lookup
        pred_ixns_embed_first = pd.DataFrame(h_data).apply(lambda row: lookup_at_nn(row, type='embed_first'), axis=1)
        pred_ixns_embed_first = torch.tensor(pred_ixns_embed_first.values)

        with open(pickle_dir+f'first_enz_embed_nn_preds_block_{args.DATA}.pickle', 'wb') as file:
            pickle.dump(pred_ixns_embed_first, file)

        # interaction predictions by FP_nn-lookup
        pred_ixns_FP = pd.DataFrame(v_data).apply(lambda row: lookup_at_nn(row, type='FP'), axis=1)
        pred_ixns_FP = torch.tensor(pred_ixns_FP.values)

        with open(pickle_dir+f'FP_nn_preds_block_{args.DATA}.pickle', 'wb') as file:
            pickle.dump(pred_ixns_FP, file)

        # interaction predictions by FP AND EC nn-lookup (diagonal set)
        pred_ixns_FP_EC = pd.DataFrame(d_data).apply(lambda row: lookup_at_nn(row, type='FP+EC'), axis=1)
        pred_ixns_FP_EC = torch.tensor(pred_ixns_FP_EC.values)

        with open(pickle_dir+f'FP+EC_nn_preds_block_{args.DATA}.pickle', 'wb') as file:
            pickle.dump(pred_ixns_FP_EC, file)

        # interaction predictions by FP AND mean_embed nn-lookup (diagonal set)
        pred_ixns_FP_mean = pd.DataFrame(d_data).apply(lambda row: lookup_at_nn(row, type='FP+mean'), axis=1)
        pred_ixns_FP_mean = torch.tensor(pred_ixns_FP_mean.values)

        with open(pickle_dir+f'FP+mean_nn_preds_block_{args.DATA}.pickle', 'wb') as file:
            pickle.dump(pred_ixns_FP_mean, file)

        # interaction predictions by FP AND first_embed nn-lookup (diagonal set)
        pred_ixns_FP_first = pd.DataFrame(d_data).apply(lambda row: lookup_at_nn(row, type='FP+first'), axis=1)
        pred_ixns_FP_first = torch.tensor(pred_ixns_FP_first.values)

        with open(pickle_dir+f'FP+first_nn_preds_block_{args.DATA}.pickle', 'wb') as file:
            pickle.dump(pred_ixns_FP_first, file)

    # test-set prediction targets (observed data)
    true_ixns_h = torch.flatten(h_data[:, -1])
    true_ixns_v = torch.flatten(v_data[:, -1])
    true_ixns_d = torch.flatten(d_data[:, -1])

    # evaluation
    evaluate(h_data, true_ixns_h, pred_ixns_ec, 'ec_nn_base_block')
    evaluate(h_data, true_ixns_h, pred_ixns_embed_mean, 'embed_mean_nn_base_block')
    evaluate(h_data, true_ixns_h, pred_ixns_embed_first, 'embed_first_nn_base_block')
    evaluate(v_data, true_ixns_v, pred_ixns_FP, 'FP_nn_base_block')
    evaluate(d_data, true_ixns_d, pred_ixns_FP_EC, 'FP+EC_nn_base_block')
    evaluate(d_data, true_ixns_d, pred_ixns_FP_mean, 'FP+mean_nn_base_block')
    evaluate(d_data, true_ixns_d, pred_ixns_FP_first, 'FP+first_nn_base_block')