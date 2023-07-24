import torch, copy, argparse, os, pickle, sys
import pandas as pd
import numpy as np

from myutil import load_block_interaction_data
from sklearn.metrics import confusion_matrix

class MLPModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, last_layer=None):
        super(MLPModel, self).__init__()

        # construct layers
        if hidden_dim == 'NO HIDDEN LAYER':
            layers = [torch.nn.ReLU(),
                      torch.nn.Dropout(dropout),
                      torch.nn.Linear(input_dim, output_dim)]
        else:
            layers = [torch.nn.Linear(input_dim, hidden_dim),
                      torch.nn.ReLU(),
                      torch.nn.Dropout(dropout),
                      torch.nn.Linear(hidden_dim, output_dim)]
        if last_layer == 'sigmoid':
            layers.append(torch.nn.Sigmoid())

        # construct model
        self.predictor = torch.nn.Sequential(*layers)

    def forward(self, X):
        return self.predictor(X)

class Recommender(torch.nn.Module):
    def __init__(self, num_cpd, num_enzyme, hidden_dim, dropout, mf_on, mlp_on, sep_embeds):
        super(Recommender, self).__init__()

        self.mf_on = mf_on
        self.mlp_on = mlp_on
        self.sep_embeds = sep_embeds
        self.embtypes = ['COMMON'] if not self.sep_embeds else [embtype for embtype in ['MF', 'MLP'] if
                                                                            getattr(self, embtype.lower() + '_on')]

        # define embeddings -> if changed, also change embedding function calls
        if self.mf_on and self.mlp_on and self.sep_embeds:
            self.MF_Embedding_Compound = torch.nn.Linear(fp_label.shape[1], hidden_dim).to(args.DEVICE)
            # self.MF_Embedding_Compound = torch.nn.Embedding(num_cpd, hidden_dim).to(args.DEVICE)

            lyrs_mf = [torch.nn.BatchNorm1d(raw_enz_embeds.shape[1], affine=False),
                       torch.nn.Linear(raw_enz_embeds.shape[1], hidden_dim)]
            self.MF_Embedding_Enzyme = torch.nn.Sequential(*lyrs_mf).to(args.DEVICE)
            # self.MF_Embedding_Enzyme = torch.nn.Linear(ec_label.shape[1], hidden_dim).to(args.DEVICE)
            # self.MF_Embedding_Enzyme = torch.nn.Embedding(num_enzyme, hidden_dim).to(args.DEVICE)

            self.MLP_Embedding_Compound = torch.nn.Linear(fp_label.shape[1], hidden_dim).to(args.DEVICE)
            # self.MLP_Embedding_Compound = torch.nn.Embedding(num_cpd, hidden_dim).to(args.DEVICE)

            lyrs_mlp = [torch.nn.BatchNorm1d(raw_enz_embeds.shape[1], affine=False),
                        torch.nn.Linear(raw_enz_embeds.shape[1], hidden_dim)]
            self.MLP_Embedding_Enzyme = torch.nn.Sequential(*lyrs_mlp).to(args.DEVICE)
            # self.MLP_Embedding_Enzyme = torch.nn.Linear(ec_label.shape[1], hidden_dim).to(args.DEVICE)
            # self.MLP_Embedding_Enzyme = torch.nn.Embedding(num_enzyme, hidden_dim).to(args.DEVICE)
        else:
            self.Embedding_Compound = torch.nn.Linear(fp_label.shape[1], hidden_dim).to(args.DEVICE)
            # self.Embedding_Compound = torch.nn.Embedding(num_cpd, hidden_dim).to(args.DEVICE)

            lyrs = [torch.nn.BatchNorm1d(raw_enz_embeds.shape[1], affine=False),
                    torch.nn.Linear(raw_enz_embeds.shape[1], hidden_dim)]
            self.Embedding_Enzyme = torch.nn.Sequential(*lyrs).to(args.DEVICE)
            # self.Embedding_Enzyme = torch.nn.Linear(ec_label.shape[1], hidden_dim).to(args.DEVICE)
            # self.Embedding_Enzyme = torch.nn.Embedding(num_enzyme, hidden_dim).to(args.DEVICE)

        # main-task network components
        if self.mlp_on:
            self.fc1 = torch.nn.Sequential(
                torch.nn.Linear(2*hidden_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim, affine=False),
                torch.nn.ReLU(),
            )

        self.dropout = torch.nn.Dropout(p=dropout)

        ce_input_dim = 2 * hidden_dim if self.mlp_on and self.mf_on else hidden_dim
        self.ce_predictor = torch.nn.Sequential(
            torch.nn.Linear(ce_input_dim, 1),
            torch.nn.Sigmoid()  # NO relu before the sigmoid!
        )

        # embedding auto-encoders
        if args.AUTOENC:
            self.fp_predictor = MLPModel(input_dim=hidden_dim, hidden_dim='NO HIDDEN LAYER',
                                         output_dim=fp_label.shape[1], dropout=dropout, last_layer='sigmoid')

            self.enz_predictor = MLPModel(input_dim=hidden_dim, hidden_dim='NO HIDDEN LAYER',
                                          output_dim=raw_enz_embeds.shape[1], dropout=dropout, last_layer='sigmoid')

        # auxiliary-task networks
        if args.KO:
            self.ko_predictor = MLPModel(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=enzyme_ko_hot.shape[1],
                                         dropout=dropout, last_layer='sigmoid')       # ko output dim is very big
        if args.EC:
            self.ec_predictor = torch.nn.ModuleList()
            for ec_dim in len_EC_fields:
                self.ec_predictor.append(MLPModel(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=ec_dim,
                                                  dropout=dropout, last_layer='sigmoid'))

    def embed(self, ids, entity, type):
        if entity == 'enzs':
            return self.embed_enz(ids, type=type)
        elif entity == 'cpds':
            return self.embed_cpd(ids, type=type)
        else:
            raise NotImplementedError("Can't embed this! Entity is neither 'cpds' nor 'enzs'.")

    def embed_enz(self, enzid, type):
        ec_enc = raw_enz_embeds[enzid].float()
        # ec_enc = ec_label[enzid].float()
        if type == 'MF':
            # return self.MF_Embedding_Enzyme(enzid)
            return self.MF_Embedding_Enzyme(ec_enc)
        elif type == 'MLP':
            # return self.MLP_Embedding_Enzyme(enzid)
            return self.MLP_Embedding_Enzyme(ec_enc)
        elif type == 'COMMON':
            # return self.Embedding_Enzyme(enzid)
            return self.Embedding_Enzyme(ec_enc)
        else:
            assert (1 == 0)

    def embed_cpd(self, cpdid, type):
        fps = fp_label[cpdid].float()
        if type == 'MF':
            # return self.MF_Embedding_Compound(cpdid)
            return self.MF_Embedding_Compound(fps)
        elif type == 'MLP':
            # return self.MLP_Embedding_Compound(cpdid)
            return self.MLP_Embedding_Compound(fps)
        elif type == 'COMMON':
            # return self.Embedding_Compound(cpdid)
            return self.Embedding_Compound(fps)
        else:
            assert (1 == 0)

    def forward(self, cpd, enz):

        # current_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
        # print(f"Current GPU memory usage: {current_memory:.2f} GB")

        # mf track
        if self.mf_on:
            embtype = 'MF' if self.sep_embeds else 'COMMON'

            emb_cpd_mf = self.embed(cpd, entity='cpds', type=embtype).float()
            emb_enz_mf = self.embed(enz, entity='enzs', type=embtype).float()

            coembed_MF = emb_enz_mf * emb_cpd_mf

        # current_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
        # print(f"Current GPU memory usage: {current_memory:.2f} GB")

        # mlp track
        if self.mlp_on:
            embtype = 'MLP' if self.sep_embeds else 'COMMON'

            if embtype == 'COMMON' and self.mf_on:  # then we already calculated the embeddings above
                emb_cpd_mlp = emb_cpd_mf
                emb_enz_mlp = emb_enz_mf
            else:
                emb_cpd_mlp = self.embed(cpd, entity='cpds', type=embtype).float()
                emb_enz_mlp = self.embed(enz, entity='enzs', type=embtype).float()

            coembed_MLP = torch.cat([emb_enz_mlp, emb_cpd_mlp], dim=-1)
            coembed_MLP = self.fc1(coembed_MLP)

        # current_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
        # print(f"Current GPU memory usage: {current_memory:.2f} GB")

        # merge tracks
        if self.mlp_on and self.mf_on:
            coembed = torch.cat([coembed_MLP, coembed_MF], dim=-1)
        elif self.mlp_on:
            coembed = coembed_MLP
        elif self.mf_on:
            coembed = coembed_MF
        else:
            assert (1 == 0)

        pred_ixn = self.ce_predictor(self.dropout(coembed))

        # current_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
        # print(f"Current GPU memory usage: {current_memory:.2f} GB")

        return pred_ixn

    # auto-encoder calls
    def predict_fp(self, compound_ids, embtype):
        embed_cpd = self.embed(compound_ids, entity='cpds', type=embtype).float()
        return self.fp_predictor(embed_cpd)

    def predict_enz_embed(self, enz_ids, embtype):
        embed_enz = self.embed(enz_ids, entity='enzs', type=embtype).float()
        return self.enz_predictor(embed_enz)

    # auxiliary tasks
    def predict_ko(self, enz_ids, embtype):
        embed_enz = self.embed(enz_ids, entity='enzs', type=embtype).float()
        return self.ko_predictor(embed_enz)

    def predict_ec(self, enz_ids, ec_i, embtype):
        embed_enz = self.embed(enz_ids, entity='enzs', type=embtype).float()
        return self.ec_predictor[ec_i](embed_enz)

def main_task_loss(output, target, alpha=1, rho=0.01, pi=0.1, form='PROB1'):

    lib = 'numpy' if isinstance(target, np.ndarray) and isinstance(output, np.ndarray) else 'torch'

    def log(x):
        return np.log(x) if lib == 'numpy' else torch.log(x)

    def clamp(x, minval, maxval):
        return np.clip(x, minval, maxval) if lib == 'numpy' else torch.clamp(x, minval, maxval)

    def neg_mean(loss):
        return -loss.mean() if lib == 'numpy' else torch.neg(torch.mean(loss))


    target, output = [i.squeeze() for i in [target, output]]  # allows faster element-wise *
    output = clamp(output, 1e-6, 1.0 - 1e-6)

    if form == 'BCE':
        loss = alpha * target * log(output) + (1 - target) * log(1 - output)
    elif form == 'PROB1':
        loss = alpha * target * log(rho * output) + (1 - target) * log(1 - output * rho)
    else:
        assert (1 == 0)

    return neg_mean(loss)


def triplet_loss(triplets, margin, entity='cpds', embtype='MF'):

    emb_anc, emb_pos, emb_neg = [model.embed(triplets[:, i], entity=entity, type=embtype) for i in range(3)]

    cos_pos, cos_neg = [torch.nn.functional.cosine_similarity(emb_anc, x, dim=-1) for x in [emb_pos, emb_neg]]

    loss = torch.clamp_min(cos_neg - cos_pos + margin, min=0.0)  # min not max bc. similarity not distance

    return loss.mean()


def contrastive_loss(anch_pos, entity='cpds', embtype='MF'):  # anch_pos = tensor([[anch,pos1],...]) on GPU
    negs = torch.randint(low=0, high=N_CPD, size=(anch_pos.shape[0],)).reshape(-1, 1).to(args.DEVICE)

    triplets = torch.cat((anch_pos, negs), dim=-1).to(args.DEVICE)

    return triplet_loss(triplets, margin=args.MARGIN, entity=entity, embtype=embtype)


def multi_task_loss(model, output, target, epoch):
    # all the ids
    cpds, enzs = [torch.arange(N).to(args.DEVICE) for N in [N_CPD, N_ENZ]]

    # main task
    main_loss = main_task_loss(output, target, alpha=args.ALPHA, rho=args.RHO, pi=args.PI, form=args.LOSS)

    # other tasks
    aux_loss = 0

    # auto-encoder tasks
    if args.AUTOENC:
        loss_fp, loss_enz = 0, 0
        for typ in model.embtypes:  # e.g. ['MF', 'MLP'] or ['COMMON']
            # fp
            fp_targets = fp_label[cpds].float()
            loss_fp += torch.nn.BCELoss()(model.predict_fp(cpds, embtype=typ), fp_targets)
            # enz
            enz_targets = raw_enz_embeds[enzs]  # .float()
            # TODO: think of a different loss (targets are not in {0,1}; BCE makes no sense)
            loss_enz += torch.nn.BCELoss()(model.predict_enz_embed(enzs, embtype=typ), enz_targets)
        aux_loss += loss_fp + loss_enz          # TODO: consider weighting, NORMALISE MAGNITUDES

    # auxiliary tasks
    if args.KO and args.EC and args.CC:
        lEC1, lEC2, lEC3 = len_EC_fields
        loss_ko, ec_losses, loss_cc = 0, [0, 0, 0], 0
        ko_targets = enzyme_ko_hot[enzs].float()    # rel large in RAM
        for typ in model.embtypes:                  # e.g. ['MF', 'MLP'] or ['COMMON']
            # enzymes: ko-task
            loss_ko += torch.nn.BCELoss()(model.predict_ko(enzs, embtype=typ), ko_targets)
            # enzymes: ec-task
            for j, ec_dim in enumerate([(0, lEC1), (lEC1, lEC1 + lEC2), (lEC1 + lEC2, lEC1 + lEC2 + lEC3)]):

                ec_target_one_hot = ec_label[enzs, ec_dim[0]:ec_dim[1]]  #.float()  #change to float for NORM and COS
                _, ec_target_class = ec_target_one_hot.max(dim=1)        # indexes of the 1s in the 1-hot encodings

                ec_pred_j = model.predict_ec(enzs, j, embtype=typ)
                ec_losses[j] += torch.nn.CrossEntropyLoss()(ec_pred_j, ec_target_class)
            # compounds: cc-task
            loss_cc += contrastive_loss(cc_anch_pos, entity='cpds', embtype=typ)

        loss_ec = sum(ec_losses)/len(ec_losses) #* 1/3  # empirical constant (make roughly same magnitude as other tasks)

        aux_loss += loss_ec + loss_ko + loss_cc  # TODO: consider weighting, NORMALISE MAGNITUDES

    # overall loss
    w_m = 1.0 if epoch > args.T else epoch / float(args.T)
    w_a = 0.0 if epoch > args.T else (1 - epoch / float(args.T))
    loss = w_m * main_loss + w_a * aux_loss

    # # only main-task loss
    # loss = main_loss

    return loss


def train(model, outfile):
    most_mean_enrichment, epoch_most_mean_enrichment, best_model_state = 0, 0, None

    print((f"{'TYPE':<5s} {'EPOCH':<5s} {'NAM':5s} {'T-LOSS':>10s} {'V-LOSS':>10s} {'MAX':>10s} {'MIN':>10s} "
           f"{'PPV':>10s} {'fREC':>10s} {'E#(+)':>10s} {'E100':>10s} {'H100':>10s}"
    ))

    for epoch in range(args.EPOCHS):
        model.train()

        # get interaction data (non-batched implementation)
        num_pos, num_neg = one_data.shape[0], zero_data.shape[0]
        neg_indices = torch.randperm(num_neg)[:args.NSR * num_pos]
        random_negatives = zero_data[neg_indices]
        train_batch = torch.cat([one_data, random_negatives], dim=0)
        obs_ixns, cpds, enzs = train_batch[:, -1].float(), train_batch[:, 0].long(), train_batch[:, 1].long()  # indices must be int64, hence long()
        # forward
        pred_ixns = model(cpds, enzs)
        # loss
        loss = multi_task_loss(model, pred_ixns, obs_ixns, epoch)
        # backprop
        opt.zero_grad()
        loss.backward()
        # print(f'GRADIENTS:\t {[p.grad for p in model.parameters() if p.requires_grad]}\n')
        opt.step()

        # eval
        if epoch % args.EVAL_FREQ == 0:
            # model selection on E100, averaged across HVAL, VVAL and DVAL
            enrich_factors = []
            for data, name in [(hval_data, 'HVAL'), (vval_data, 'VVAL'), (dval_data, 'DVAL')]:
                mtl, enrich_100 = evaluate(model, data, loss, outfile, epoch=epoch, type=name)
                enrich_factors.append(enrich_100)

            mean_enrichment = torch.tensor(enrich_factors).float().mean().item()
            median_enrichment = torch.tensor(enrich_factors).float().median().item()
            print(f'AVERAGE & MEDIAN ENRICHMENT TOP 100: {mean_enrichment}, {median_enrichment}\n')

            if mean_enrichment > most_mean_enrichment:
                most_mean_enrichment, epoch_most_mean_enrichment = mean_enrichment, epoch
                best_model_state = copy.deepcopy(model.state_dict())

        # # freeze embedding params after a while
        # if epoch == args.T:
        #     print(f'\n FREEZING EMB WEIGHTS FROM EPOCH {epoch_best_auprc}')
        #     model.load_state_dict(best_model_state)
        #     if model.sep_embeds:
        #         components_to_freeze = [model.MF_Embedding_Compound.parameters(),
        #                                 model.MF_Embedding_Enzyme.parameters(),
        #                                 model.MLP_Embedding_Compound.parameters(),
        #                                 model.MLP_Embedding_Enzyme.parameters()]
        #     else:
        #         components_to_freeze = [model.Embedding_Compound.parameters(),
        #                                 model.Embedding_Enzyme.parameters()]
        #     for component in components_to_freeze:
        #         for param in component:
        #             param.requires_grad = False

    print(f'\nFINISHED TRAINING -- TESTING BEST MODEL (from epoch {epoch_most_mean_enrichment})')
    print((f"{'TYPE':<5s} {'EPOCH':<5s} {'NAM':5s} {'T-LOSS':>10s} {'V-LOSS':>10s} {'MAX':>10s} {'MIN':>10s} "
           f"{'PPV':>10s} {'fREC':>10s} {'E#(+)':>10s} {'E100':>10s} {'H100':>10s}"
    ))
    model.load_state_dict(best_model_state)
    datas = [dte_data, hte_data, vte_data, h_va_te_data, v_va_te_data]
    names = ['DTEST', 'HTEST', 'VTEST', 'HVTEST', 'VVEST']

    colnames = ['TYPE', 'V-LOSS', 'PPV', 'fRec', 'E#(+)', 'E100', 'H100']
    df = pd.DataFrame(columns=colnames)
    for data, name in zip(datas, names):
        mtl, ppv, frec, enrich_npos, enrich_100, yield_100 = evaluate(model, data, loss, outfile, epoch=-1, type=name)
        df.loc[len(df)] = (name, mtl, ppv, frec, enrich_npos, enrich_100, yield_100)

    df.to_csv("".join([cwd, '/', args.OUT, '/', outfile, f'_block_{args.DATA}_run_{args.REPL}.csv']), index=False)


def evaluate(model, data, last_train_loss, outfile, epoch=0, type='DVAL', **kwargs):
    with torch.no_grad():
        model.eval()

        # interaction prediction targets (observed data)
        true_ixns = torch.flatten(data[:, -1])

        # interaction predictions by model
        pred_ixns = torch.zeros_like(true_ixns, dtype=torch.float)
        batch_size = 20480  # batches for memory-efficient forward-call
        for bi in range(int(np.ceil(data.shape[0] / batch_size))):
            indices_s, indices_e = bi * batch_size, min(data.shape[0], (bi + 1) * batch_size)
            cpds, enzs = [data[indices_s:indices_e, i].long() for i in [0, 1]]
            pred_ixns[indices_s:indices_e] = torch.flatten(model(cpds, enzs))

        if epoch == 0:
            # also do baseline predictions
            all_zero, all_ones = torch.zeros_like(true_ixns), torch.ones_like(true_ixns)
            unif_rand = torch.rand_like(all_zero.float())                # uniformly random on [0,1]
            prior_rand = torch.where(unif_rand > 1 - args.PI, 1.0, 0.0)  # predict 1 with prob args.pi
            # convert prediction vectors to numpy, place on CPU
            ypreds = [y.cpu().numpy() for y in [pred_ixns, all_zero, all_ones, unif_rand, prior_rand]]
            eval_list = zip(['MODEL', 'ZEROS', ' ONES', ' UNIF', 'PRIOR'], ypreds)
        else:
            eval_list = [ ( 'MODEL', pred_ixns.cpu().numpy() ) ]
        true_ixns = true_ixns.cpu().numpy()

        for name, ypred in eval_list:
            # no-threshold metrics (including validation loss)
            mtl = main_task_loss(ypred, true_ixns, alpha=args.ALPHA, rho=args.RHO, pi=args.PI, form=args.LOSS)
            maxpred, minpred = max(ypred), min(ypred)

            # threshold-specific metrics
            ypred_bin_5 = np.where(ypred > 0.5, 1.0, 0.0)  # binarise predictions with threshold = 0.5
            tn, fp, fn, tp = confusion_matrix(true_ixns, ypred_bin_5).ravel()

            ppv = np.nan if name == 'ZEROS' else tp / (tp + fp)   # positive predictive value (= precision); lower bound wrt unknown ground truth
            frec = tp / (tp + fn)  # fraction of observed (+) recovered; no specific relationship to ground-truth TPR

            # enrichment scores
            num_obs_pos = int(true_ixns.sum())  # number of observed (+) interactions
            posrate = num_obs_pos / true_ixns.shape[0]  # chance to get a hit at random (in 1 draw)

            sorted_pred_indices = np.argsort(ypred)[::-1]  # predictions DEscending by logits
            indices = [sorted_pred_indices[:i] for i in [num_obs_pos, 100]]  # top #observed(+), 100 preds
            yield_npos, yield_100 = [int(true_ixns[index].sum()) for index in indices]  # hits in those predictions

            enrich_npos = yield_npos / (num_obs_pos * posrate)  # enrichment relative to random guessing
            enrich_100 = yield_100 / (100 * posrate)

            # output
            if name != 'MODEL':
                last_train_loss = np.nan

            print((f"{type:5s} {epoch:<5d} {name:5s} {last_train_loss:>10.4f} {mtl:>10.4f} "
                   f"{maxpred:>10.4f} {minpred:>10.4f} {ppv:>10.3f} {frec:>10.4f} "
                   f"{enrich_npos:>10.1f} {enrich_100:>10.1f} {yield_100:>10d}"
            ))

            returnvals = (mtl, enrich_100)   # will be returned, unless last epoch

            if epoch == -1:
                returnvals = (mtl, ppv, frec, enrich_npos, enrich_100, yield_100)

                # save correct hit pairs
                def save_hits(top_pred_indices):
                    top_pred_data = data[top_pred_indices.copy()]  # copy-workaround for np limitation (no neg strides!)
                    hit_rows = top_pred_data[top_pred_data[:, -1] == 1]

                    out = pd.DataFrame(hit_rows.cpu(), columns=['substrate', 'enzyme', 'observed interaction'])

                    out['substrate'] = out['substrate'].apply(lambda s_id: substrate_key[s_id])
                    out['enzyme'] = out['enzyme'].apply(lambda e_id: enzyme_key[e_id])

                    out.to_csv(
                        "".join([cwd, '/', args.OUT, '/', outfile, f'block_{args.DATA}_run_{args.REPL}_hits{len(top_pred_indices)}.txt']),
                        index=False)

                if name == 'MODEL':
                    save_hits(indices[1])  # correct hits from top 100 predictions

        if epoch == 0: print('\n')

    return returnvals

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Task Learning on Real System with Seq Embeddings")
    # system and files
    parser.add_argument('--DEVICE', type=str, default='cuda')
    parser.add_argument('--DATA', type=int, default=0)                      # which block split dataset
    parser.add_argument('--REPL', type=int, default=0)                      # which replicate of the same block?
    parser.add_argument('--OUT', type=str, default='outdir')
    parser.add_argument('--OUTF', type=str, default='outfile')

    # training parameters
    parser.add_argument('--EPOCHS', type=int, default=3500)
    parser.add_argument('--LR', type=float, default=0.001)
    parser.add_argument('--L2', type=float, default=1e-6)
    parser.add_argument('--DROPOUT', type=float, default=0.5)
    parser.add_argument('--PI', type=float, default=0.01)                   # assumed true fraction of (+) ixns
    parser.add_argument('--RHO', type=float, default=0.1)                   # of these, fraction that are observed
    parser.add_argument('--ALPHA', type=float, default=1)                   # assym. penalty for FN
    parser.add_argument('--NSR', type=float, default=1)                     # negative sampling ratio (+):(-) = 1:NSR
    parser.add_argument('--LOSS', choices=['BCE', 'PROB1'], default='BCE')  # main task loss
    parser.add_argument('--T', type=float, default=2000)                    # loss mixing schedule
    parser.add_argument('--MARGIN', type=float, default=1)                  # for triplet loss
    parser.add_argument('--EVAL_FREQ', type=int, default=50)                # in epochs
    # parser.add_argument('--early_stop_window', type=int, default=200)     # should be multiple of eval_freq

    # model structure
    parser.add_argument('--DIM', type=int, default=2048)                     # learnt-embedding dimension
    parser.add_argument('--MF_on', action='store_true')                     # MF track of NCF network
    parser.add_argument('--MF_off', dest='MF_on', action='store_false')
    parser.add_argument('--MLP_on', action='store_true')                    # MLP track of NCF network
    parser.add_argument('--MLP_off', dest='MLP_on', action='store_false')
    parser.add_argument('--SEP_EMBEDS', action='store_true')                # separate or joint embeddings for {MF,MLP}
    parser.add_argument('--NO_SEP_EMBEDS', dest='SEP_EMBEDS', action='store_false')

    # tasks
    parser.add_argument('--AUTOENC', action='store_true')     # FP & enz_embed autoencoder tasks ON
    parser.add_argument('--KO', action='store_true')          # enz_KO prediction task ON
    parser.add_argument('--EC', action='store_true')          # enz_EC prediction task ON
    parser.add_argument('--CC', action='store_true')          # cpd_CC prediction task ON

    parser.set_defaults(MF_on=True, MLP_on=True, SEP_EMBEDS=True,
                                    AUTOENC=False, KO=True, EC=True, CC=True)  # TODO: rethink EMBED_AUTOENC LOSS (NOT BCE!)
    args = parser.parse_args()

    if args.SEP_EMBEDS: assert (args.MF_on and args.MLP_on)

    # global definitions
    # torch.set_num_threads(4)
    # system
    cwd = os.getcwd()                     # working directory
    os.makedirs(args.OUT, exist_ok=True)  # output directory

    with open(args.OUT+'/'+args.OUTF+f"block_{args.DATA}_run_{args.REPL}", 'w') as file:
        sys.stdout = file

        print('\n---- SYSTEM PARAMETERS ----')
        print(f'GPU:\t\t\t {torch.cuda.is_available()}')
        print(f'DATA:\t\t\t block_{args.DATA}')
        print(f'REPL:\t\t\t run_{args.REPL}')
        print('\n---- TRAINING PARAMETERS ----')
        print(f'NUM EPOCHS:\t\t {args.EPOCHS}')
        print(f'LEARN RATE:\t\t {args.LR}')
        print(f'L2 REG:\t\t\t {args.L2}')
        print(f'DROPOUT:\t\t {args.DROPOUT}')
        print(f'PI:\t\t\t {args.PI}')
        print(f'RHO:\t\t\t {args.RHO}')
        print(f'ALPHA:\t\t\t {args.ALPHA}')
        print(f'NSR:\t\t\t {args.NSR}')
        print(f'MAIN LOSS:\t\t\t {args.LOSS}')
        print(f'T:\t\t\t {args.T}')
        print(f'EVAL FREQ:\t\t {args.EVAL_FREQ} epoch(s)')

        # enzyme and substrate entities: {internal_id : EC or KEGG compound}
        with open('/cluster/home/ovavourakis/projects/src/data/block_splits/substrate_key.pkl', 'rb') as fi:
            substrate_key = pickle.load(fi)
        with open('/cluster/home/ovavourakis/projects/src/data/block_splits/enzyme_key.pkl', 'rb') as fi:
            enzyme_key = pickle.load(fi)

        # interaction data
        file = f'/cluster/home/ovavourakis/projects/src/data/block_splits/interaction_real_cofacs-train_block_{args.DATA}.pkl'
        # file = f'/Users/odysseasvavourakis/Documents/2022-2023/Studium/5. Semester/Thesis Work/src/boost-rs-repl/data/interaction_real_after_toy_block2.pkl'
        tr, d_va, h_va, v_va, d_te, h_te, v_te, h_va_te, v_va_te, \
            N_CPD, N_ENZ, fp_label, ec_label, len_EC_fields, \
            EC_to_hot_dicts, hot_to_EC_dicts, pos_to_ko_dict, ko_to_pos_dict, \
            num_ko, CC_dict, enzyme_ko_hot = load_block_interaction_data(file)
        ixn_datasets = [d.to(args.DEVICE) for d in [tr, d_va, h_va, v_va, d_te, h_te, v_te, h_va_te, v_va_te]]
        train_data, dval_data, hval_data, vval_data, dte_data, hte_data, vte_data, h_va_te_data, v_va_te_data = ixn_datasets
        # split train into (+) and (-)
        one_idx, zero_idx = [ torch.where(train_data[:, 2] == i)[0] for i in [1,0] ]
        one_data, zero_data = [ train_data[idx].to(args.DEVICE) for idx in [one_idx, zero_idx] ]

        # enz embeddings
        # file = f'/cluster/project/krause/ovavourakis/boost-rs-repl/data/seq_embeds/esm2_3B_mean.pkl'
        file = f'/cluster/project/krause/ovavourakis/boost-rs-repl/data/seq_embeds/esm2_3B_first.pkl'
        with open(file, 'rb') as f:
            raw_enz_embeds = pickle.load(f).to(args.DEVICE)
            print('\nENZYME EMBEDDINGS')
            print(file)
            print('\n')

        # cpd embeddings
        fp_label = fp_label.to(args.DEVICE)

        # auxiliary data:  (ko, ec, cc)-tasks
        cc_anch_pos = torch.tensor([(anch, pos) for anch, poses in CC_dict.items() for pos in poses], dtype=torch.int)
        aux_targets = [enzyme_ko_hot.int(), ec_label, cc_anch_pos]
        enzyme_ko_hot, ec_label, cc_anch_pos = [d.to(args.DEVICE) for d in aux_targets]

        print('\n---- DATA ----')
        names = ["train_data", "dval_data", "hval_data", "vval_data", "dte_data",
                                            "hte_data", "vte_data", "h_va_te_data", "v_va_te_data"]
        for name, d in zip(names, ixn_datasets):
            print(name, d[:, -1].sum().item() , '/', d.shape[0], '=', d[:, -1].sum().item() / d.shape[0])

        # construct model
        model = Recommender(num_cpd=N_CPD, num_enzyme=N_ENZ, hidden_dim=args.DIM, dropout=args.DROPOUT,
                            mf_on=args.MF_on, mlp_on=args.MLP_on, sep_embeds=args.SEP_EMBEDS).to(args.DEVICE)

        opt = torch.optim.AdamW(model.parameters(), lr=args.LR, weight_decay=args.L2)

        print('\n---- MODEL STRUCTURE ----')
        print(f'EMBED DIM:\t {args.DIM}')
        print(f'MF_track:\t {model.mf_on}')
        print(f'MLP_track:\t {model.mlp_on}')
        print(f'SEPARATE EMBEDS: {model.sep_embeds}')
        print(f'TOTAL PARAMS:\t {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n')

        print('\n---- TASKS ----')
        print(f'AUTOENC: {args.AUTOENC}')
        print(f'EC: {args.EC}')
        print(f'KO: {args.KO}')
        print(f'CC: {args.CC}')

        # current_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
        # print(f"Current GPU memory usage: {current_memory:.2f} GB")

        train(model, args.OUTF)