import torch, copy, argparse, os, pickle, sys
import pandas as pd
import numpy as np

from tqdm import tqdm

from modules import TransformerLayer, ESM1bLayerNorm
from myutil import load_block_interaction_data
from sklearn.metrics import confusion_matrix

from torch.nn.parallel import DataParallel
from torch.nn.parallel.scatter_gather import scatter_kwargs, scatter
from parallel_apply import parallel_apply_embed, parallel_apply_ko, parallel_apply_ec

class MyDataParallel(DataParallel):
    def enz_aux_tasks(self, indices, type):
        # scatter indices across GPUs
        inputs = scatter(indices, self.device_ids)
        inputs = [(raw_enz_embeds[idx, :, :512].to(device),
                   padding_masks[idx, :].to(device))
                  for idx, device in zip(inputs, self.device_ids)]
        kwargs = [{'type': type} for device in self.device_ids]
        # replicate model across devices
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        # embed enzymes in parallel over GPUs
        embeds = parallel_apply_embed(replicas, inputs, kwargs)
        # ko prediction in parallel over GPUs
        outputs_ko = parallel_apply_ko(replicas, embeds)
        outputs_ko = self.gather(outputs_ko, self.output_device)
        # ec predictions in parallel over GPUs
        outputs_ec = []
        for i in range(3):
            kwargs = [{'ec_i': i} for device in self.device_ids]  # which ec_i (1,2, or 3)
            outputs = parallel_apply_ec(replicas, embeds, kwargs)
            outputs = self.gather(outputs, self.output_device)
            outputs_ec.append(outputs)

        return outputs_ko, outputs_ec

    def forward(self, indices):
        '''
        :param indices: tensor of shape (batch size, 2); 1st col is cpd_ids, 2nd col is enz_ids
        '''

        if not self.device_ids:
            return self.module(indices)

        # split the indices across GPUs
        # inputs, kwargs = scatter_kwargs((indices,) {}, self.device_ids)
        inputs = scatter(indices, self.device_ids)

        # load the appropriate embedding slices onto each GPU # TODO: change enz dimension (also above)
        inputs = [ ( fp_label[idx[:, 0]].to(device),
                     raw_enz_embeds[idx[:, 1], :, :512].to(device),
                     padding_masks[idx[:, 1], :].to(device)         )
                    for idx, device in zip(inputs, self.device_ids)]

        if len(self.device_ids) == 1:
            return self.module(*inputs[0], None)
        else:
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
            outputs = self.parallel_apply(replicas, inputs, None)   # no kwargs
            return self.gather(outputs, self.output_device)


class Transformer(torch.nn.Module):
    # embedding dim should be multiple of 64 and num attention heads edim/64
    # def __init__(self, hidden_dim, trafo_embed_dim=2560, num_attention_heads=1):
    def __init__(self, hidden_dim, trafo_embed_dim=512, num_attention_heads=8):     # TODO: change dim -> also down where you load them
        super(Transformer, self).__init__()

        self.trafo_embed_dim = trafo_embed_dim
        self.num_attention_heads = num_attention_heads

        self.trafo = TransformerLayer(trafo_embed_dim,             # input and output embedding dim
                                      trafo_embed_dim,             # intermediate dim in feed-forward part
                                      num_attention_heads,
                                      add_bias_kv=False, use_esm1b_layer_norm=True, use_rotary_embeddings=True)

        self.emb_layer_norm_after = ESM1bLayerNorm(self.trafo_embed_dim)

        self.feed_forward = torch.nn.Linear(trafo_embed_dim, hidden_dim)

    def forward(self, hidden_matrix, padding_mask):
        '''
        :param hidden_matrix:   (batch size, sequence length, embedding dimension)
        :param padding_mask:    (batch size, sequence length)
        :return:                (batch size, embedding dimension)
        '''

        hidden_matrix = hidden_matrix.transpose(0, 1)       # sequence length, batch size, dimension
        hidden_matrix = self.trafo(hidden_matrix,
                                   self_attn_padding_mask=padding_mask)[0]   # return hidden matrix, not attn weights
        hidden_matrix = self.emb_layer_norm_after(hidden_matrix)
        hidden_matrix = hidden_matrix.transpose(0, 1)       # batch size, sequence length, dimension

        enzyme_embedding_vector = hidden_matrix[:, 1, :]    # take first column of each matrix
        enzyme_embedding_vector = self.feed_forward(enzyme_embedding_vector)

        return enzyme_embedding_vector

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
            self.MF_Embedding_Compound = torch.nn.Linear(fp_label.shape[1], hidden_dim)#.to(args.DEVICE)
            self.MF_Embedding_Enzyme = Transformer(hidden_dim)#.to(args.DEVICE)

            self.MLP_Embedding_Compound = torch.nn.Linear(fp_label.shape[1], hidden_dim).to(args.DEVICE)
            self.MLP_Embedding_Enzyme = Transformer(hidden_dim).to(args.DEVICE)
        else:
            self.Embedding_Compound = torch.nn.Linear(fp_label.shape[1], hidden_dim).to(args.DEVICE)
            self.Embedding_Enzyme = Transformer(hidden_dim).to(args.DEVICE)

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

        # auxiliary-task networks
        if args.KO:
            self.ko_predictor = MLPModel(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=enzyme_ko_hot.shape[1],
                                         dropout=dropout, last_layer='sigmoid')       # ko output dim is very big
        if args.EC:
            self.ec_predictor = torch.nn.ModuleList()
            for ec_dim in len_EC_fields:
                self.ec_predictor.append(MLPModel(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=ec_dim,
                                                  dropout=dropout, last_layer='sigmoid'))

    # def embed(self, raw_embeds, padding, entity='cpds', type='MF'):
    #     if entity == 'enzs':
    #         return self.embed_enz(raw_embeds, padding, type=type)
    #     elif entity == 'cpds':
    #         return self.embed_cpd(raw_embeds, type=type)
    #     else:
    #         raise NotImplementedError("Can't embed this! Entity is neither 'cpds' nor 'enzs'.")

    def embed_enz(self, h, mask, type='MF'):
        # # read hidden matrices from file
        # print('reading hiddens from file')
        # print(enzid.device)
        # print(enzid)
        # # h = []
        # # for id in enzid:
        #     print(enzid.device,f'reading hidden matrix {id}')
        #     file = f'/cluster/scratch/ovavourakis/embeds/single_3B/hidden_{id}.pkl'
        #     with open(file, 'rb') as f:
        #         hi = pickle.load(f)[:, :100]        # maxlen, trafo_embed_dim
        #     f.close()
        #     # hi = pickle.load(f)                                # maxlen, trafo_embed_dim
        #     h.append(hi.unsqueeze(0))
        # h = torch.cat(h, dim=0).to(enzid.device)     # batch size, maxlen, trafo_embed_dim

        # current_memory = torch.cuda.memory_allocated(device = h.device) / (1024 ** 3)  # Convert bytes to GB
        # print(f"Current GPU memory usage: {current_memory:.2f} GB")

        # run through transformer
        if type == 'MF':
            return self.MF_Embedding_Enzyme(h, mask)
        elif type == 'MLP':
            return self.MLP_Embedding_Enzyme(h, mask)
        elif type == 'COMMON':
            return self.Embedding_Enzyme(h, mask)
        else:
            assert (1 == 0)

    def embed_cpd(self, raw_embeds, type):
        if type == 'MF':
            return self.MF_Embedding_Compound(raw_embeds.float())
        elif type == 'MLP':
            return self.MLP_Embedding_Compound(raw_embeds.float())
        elif type == 'COMMON':
            return self.Embedding_Compound(raw_embeds.float())
        else:
            assert (1 == 0)

    def forward(self, input_embeds_cpd, input_embeds_enz, enz_padding, *args, **kwargs):
        # mf track
        if self.mf_on:
            embtype = 'MF' if self.sep_embeds else 'COMMON'

            emb_cpd_mf = self.embed_cpd(input_embeds_cpd, type=embtype)#.float()
            emb_enz_mf = self.embed_enz(input_embeds_enz, enz_padding, type=embtype)#.float()

            coembed_MF = emb_enz_mf * emb_cpd_mf

        # mlp track
        if self.mlp_on:
            embtype = 'MLP' if self.sep_embeds else 'COMMON'

            if embtype == 'COMMON' and self.mf_on:  # then we already calculated the embeddings above
                emb_cpd_mlp = emb_cpd_mf
                emb_enz_mlp = emb_enz_mf
            else:
                emb_cpd_mlp = self.embed_cpd(input_embeds_cpd, type=embtype)  # .float()
                emb_enz_mlp = self.embed_enz(input_embeds_enz, enz_padding, type=embtype)  # .float()

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

    # auxiliary tasks
    def predict_ko(self, embed_enz):#, padding, embtype):
        return self.ko_predictor(embed_enz)

    def predict_ec(self, embed_enz, ec_i=1):
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


def triplet_loss(triplets, margin, embtype='MF'):
    inp_embs = [fp_label[cpdids].float() for cpdids in [triplets[:, i] for i in range(3)]]

    emb_anc, emb_pos, emb_neg = [model.module.embed_cpd(inp_emb, type=embtype) for inp_emb in inp_embs]

    cos_pos, cos_neg = [torch.nn.functional.cosine_similarity(emb_anc, x, dim=-1) for x in [emb_pos, emb_neg]]

    loss = torch.clamp_min(cos_neg - cos_pos + margin, min=0.0)  # min not max bc. similarity not distance

    return loss.mean()


def contrastive_loss(anch_pos, embtype='MF'):  # anch_pos = tensor([[anch,pos1],...]) on GPU
    negs = torch.randint(low=0, high=N_CPD, size=(anch_pos.shape[0],)).reshape(-1, 1).to(args.DEVICE)

    triplets = torch.cat((anch_pos, negs), dim=-1).to(args.DEVICE)

    return triplet_loss(triplets, margin=args.MARGIN, embtype=embtype)


def multi_task_loss(model, output, target, epoch):
    enzs = torch.randperm(N_ENZ)[:500].to(args.DEVICE)  # 500 random enzyme ids for aux tasks
    cpds = torch.arange(N_CPD).to(args.DEVICE)          # all the compounds for aux task

    # main task
    main_loss = main_task_loss(output, target, alpha=args.ALPHA, rho=args.RHO, pi=args.PI, form=args.LOSS)

    # auxiliary tasks
    aux_loss = 0
    if args.KO and args.EC and args.CC:
        lEC1, lEC2, lEC3 = len_EC_fields
        loss_ko, ec_losses, loss_cc = 0, [0, 0, 0], 0
        ko_targets = enzyme_ko_hot[enzs].float()    # rel large in RAM
        for typ in model.module.embtypes:                  # e.g. ['MF', 'MLP'] or ['COMMON']
            # enzymes: ko-task & ec-task
            preds_ko, preds_ec = model.enz_aux_tasks(enzs, typ)
            loss_ko += torch.nn.BCELoss()(preds_ko, ko_targets)
            for j, ec_dim in enumerate([(0, lEC1), (lEC1, lEC1 + lEC2), (lEC1 + lEC2, lEC1 + lEC2 + lEC3)]):

                ec_target_one_hot = ec_label[enzs, ec_dim[0]:ec_dim[1]]  #.float()  #change to float for NORM and COS
                _, ec_target_class = ec_target_one_hot.max(dim=1)        # indexes of the 1s in the 1-hot encodings

                ec_losses[j] += torch.nn.CrossEntropyLoss()(preds_ec[j] , ec_target_class)
            # compounds: cc-task
            loss_cc += contrastive_loss(cc_anch_pos, embtype=typ)

        loss_ec = sum(ec_losses)/len(ec_losses) #* 1/3  # empirical constant (make roughly same magnitude as other tasks)

        aux_loss += loss_ec + loss_ko + loss_cc  # TODO: consider weighting, NORMALISE MAGNITUDES

    # overall loss
    if aux_loss != 0:
        w_m = 1.0 if epoch > args.T else epoch / float(args.T)
        w_a = 0.0 if epoch > args.T else (1 - epoch / float(args.T))
        loss = w_m * main_loss + w_a * aux_loss
    else:
        loss = main_loss

    return loss


def train(model, outfile):
    most_mean_enrichment, epoch_most_mean_enrichment, best_model_state = 0, 0, None

    print((f"{'TYPE':<5s} {'EPOCH':<5s} {'NAM':5s} {'T-LOSS':>10s} {'V-LOSS':>10s} {'MAX':>10s} {'MIN':>10s} "
           f"{'PPV':>10s} {'fREC':>10s} {'E#(+)':>10s} {'E100':>10s} {'H100':>10s}"
    ))

    for epoch in range(args.EPOCHS):
        print(f'EPOCH {epoch}')

        model.train()

        # implementation with gradient aggregation working through positive pairs
        num_pos, num_neg = one_data.shape[0], zero_data.shape[0]
        batch_size = 250    # number of positive examples in batch
        num_batches = int(np.ceil(num_pos / batch_size))
        for bi in tqdm(range(num_batches)):
            # get the next batch of positive pairs
            indices_s, indices_e = bi * batch_size, min(num_pos, (bi + 1) * batch_size)
            pos_pairs = one_data[indices_s:indices_e, :]
            # get some number of negative pairs
            neg_indices = torch.randperm(num_neg)[:args.NSR * batch_size]
            random_neg_pairs = zero_data[neg_indices, :]
            # create training batch (and mix + and -)
            train_batch = torch.cat([pos_pairs, random_neg_pairs], dim=0)
            random_indices = torch.randperm(train_batch.shape[0])
            train_batch = train_batch[random_indices, :]
            obs_ixns, cpds_enzs = train_batch[:, -1].float(), train_batch[:, :2].long()  # long(): indices must be int64
            # forward
            pred_ixns = model(cpds_enzs)
            # loss
            loss = multi_task_loss(model, pred_ixns, obs_ixns, epoch) #/ num_batches
            # backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

        # # get random subset of interaction data (5 positives + 5*NSR negatives)
        # print('train(): get data')
        # num_pos, num_neg = one_data.shape[0], zero_data.shape[0]
        #
        # random_indices = torch.randperm(num_pos)[:5]
        # random_pos_pairs = one_data[random_indices]
        #
        # neg_indices = torch.randperm(num_neg)[:args.NSR * 5]
        # random_neg_pairs = zero_data[neg_indices]
        #
        # train_batch = torch.cat([random_pos_pairs, random_neg_pairs], dim=0)
        # obs_ixns, cpds_enzs = train_batch[:, -1].float(), train_batch[:, :2].long()  # indices must be int64, hence long()
        # # forward
        # print('train(): fwd pass')
        # pred_ixns = model(cpds_enzs)
        # # loss
        # print('train(): loss comp')
        # loss = multi_task_loss(model, pred_ixns, obs_ixns, epoch)
        # # backprop
        # print('train(): bwd pass')
        # opt.zero_grad()
        # loss.backward()
        # # print(f'GRADIENTS:\t {[p.grad for p in model.parameters() if p.requires_grad]}\n')
        # opt.step()

        # eval
        if epoch % args.EVAL_FREQ == 0 :#and epoch != 0:
            # model selection on E100, averaged across HVAL, VVAL and DVAL
            enrich_factors = []
            eval_datasets = [(one_data_hval, zero_data_hval), (one_data_vval, zero_data_vval),
                             (one_data_dval, zero_data_dval)]
            for data, name in zip(eval_datasets, ['HVAL', 'VVAL', 'DVAL']):
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

        # interaction predictions by model (sampling negatives)
        if len(data) == 2:          # if we have one_data and zero_data separately
            odata, zdata = data
            num_pos, num_neg = odata.shape[0], zdata.shape[0]
            batch_size = 100  # number of positive examples in batch
            num_batches = int(np.ceil(num_pos / batch_size))
            pred_ixns, true_ixns = [], []
            print('EVALUATING')
            for bi in tqdm(range(num_batches)):
                # get the next batch of positive pairs
                indices_s, indices_e = bi * batch_size, min(num_pos, (bi + 1) * batch_size)
                pos_pairs = odata[indices_s:indices_e, :]
                # get some number of negative pairs
                neg_indices = torch.randperm(num_neg)[:19 * batch_size]
                random_neg_pairs = zdata[neg_indices, :]
                # create eval batch
                eval_batch = torch.cat([pos_pairs, random_neg_pairs], dim=0)
                obs_ixns, cpds_enzs = eval_batch[:, -1].float(), eval_batch[:, :2].long()  # long(): indices must be int64
                true_ixns.append(obs_ixns)
                # forward
                pred_ixns.append(model(cpds_enzs))
            pred_ixns, true_ixns = [torch.flatten(torch.cat(d).float()) for d in [pred_ixns, true_ixns]]
        else:
            # interaction prediction targets (observed data)
            true_ixns = torch.flatten(data[:, -1])

            # interaction predictions by model (as-is)
            pred_ixns = torch.zeros_like(true_ixns, dtype=torch.float)
            batch_size = 2000#0  # batches for memory-efficient forward-call
            num_batches = int(np.ceil(data.shape[0] / batch_size))
            print('EVALUATING')
            for bi in tqdm(range(num_batches)):
                indices_s, indices_e = bi * batch_size, min(data.shape[0], (bi + 1) * batch_size)
                eval_batch = data[indices_s:indices_e, :2].long()
                pred_ixns[indices_s:indices_e] = torch.flatten(model(eval_batch))

        if epoch == 0:
            # also do baseline predictions
            all_zero, all_ones = torch.zeros_like(true_ixns).float(), torch.ones_like(true_ixns).float()
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
            ypred_bin_5 = np.where(ypred > 0.5, 1.0, 0.0) # binarise predictions with threshold = 0.5
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
                def save_hits(top_pred_indices, type):
                    top_pred_data = data[top_pred_indices.copy()]  # copy-workaround for np limitation (no neg strides!)
                    hit_rows = top_pred_data[top_pred_data[:, -1] == 1]

                    out = pd.DataFrame(hit_rows.cpu(), columns=['substrate', 'enzyme', 'observed interaction'])

                    out['substrate'] = out['substrate'].apply(lambda s_id: substrate_key[s_id.item()])
                    out['enzyme'] = out['enzyme'].apply(lambda e_id: enzyme_key[e_id.item()])

                    out.to_csv(
                        "".join([cwd, '/', args.OUT, '/', outfile, f'block_{args.DATA}_run_{args.REPL}_hits{len(top_pred_indices)}_{type}.txt']),
                        index=False)

                if name == 'MODEL':
                    save_hits(indices[1], type)  # correct hits from top 100 predictions

        if epoch == 0: print('\n')

    return returnvals

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Task Learning on Real System with Matrix Seq Embeddings")
    # system and files
    parser.add_argument('--DEVICE', type=str, default='cuda')
    parser.add_argument('--DATA', type=int, default=0)                      # which block split dataset
    parser.add_argument('--REPL', type=int, default=0)                      # which replicate of the same block?
    parser.add_argument('--OUT', type=str, default='outdir')
    parser.add_argument('--OUTF', type=str, default='outfile')

    # training parameters
    parser.add_argument('--EPOCHS', type=int, default=20)#3500)                # TODO: reconsider and also <-> args.T
    parser.add_argument('--LR', type=float, default=0.001)
    parser.add_argument('--L2', type=float, default=1e-6)
    parser.add_argument('--DROPOUT', type=float, default=0.5)
    parser.add_argument('--PI', type=float, default=0.01)                   # assumed true fraction of (+) ixns
    parser.add_argument('--RHO', type=float, default=0.1)                   # of these, fraction that are observed
    parser.add_argument('--ALPHA', type=float, default=1)                   # assym. penalty for FN
    parser.add_argument('--NSR', type=float, default=1)                     # TODO: reconsider # negative sampling ratio (+):(-) = 1:NSR
    parser.add_argument('--LOSS', choices=['BCE', 'PROB1'], default='BCE')  # main task loss
    parser.add_argument('--T', type=float, default=10)#700)                    # TODO: reconsider # loss mixing schedule
    parser.add_argument('--MARGIN', type=float, default=1)                  # for triplet loss
    parser.add_argument('--EVAL_FREQ', type=int, default=1)             # TODO: don't forget to cgange ritical T
    # parser.add_argument('--early_stop_window', type=int, default=200)     # should be multiple of eval_freq

    # model structure
    parser.add_argument('--DIM', type=int, default=512)                     # TODO: reconsider # learnt-embedding dimension
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
                                    AUTOENC=False, KO=True, EC=True, CC=True)  # TODO: reconsider
    args = parser.parse_args()

    if args.SEP_EMBEDS: assert (args.MF_on and args.MLP_on)

    # global definitions
    # torch.set_num_threads(4)
    # system
    cwd = os.getcwd()                     # working directory
    os.makedirs(args.OUT, exist_ok=True)  # output directory
    with open(args.OUT+'/'+args.OUTF+f"block_{args.DATA}_run_{args.REPL}", 'w') as file:
        sys.stdout = file                                     # TODO: re-enable

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
        # print(f'EARLY STOP WINDOW:\t {args.early_stop_window} epoch(s)')

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
        # ixn_datasets = [d.to(args.DEVICE) for d in [tr, d_va, h_va, v_va, d_te, h_te, v_te, h_va_te, v_va_te]]
        ixn_datasets = [tr, d_va, h_va, v_va, d_te, h_te, v_te, h_va_te, v_va_te]
        train_data, dval_data, hval_data, vval_data, dte_data, hte_data, vte_data, h_va_te_data, v_va_te_data = ixn_datasets
        for d in [d_te, h_te, v_te, h_va_te, v_va_te]:
            d = d.to(args.DEVICE)
        # split train into (+) and (-)
        one_idx, zero_idx = [ torch.where(train_data[:, 2] == i)[0] for i in [1, 0] ]
        one_data, zero_data = [ train_data[idx].to(args.DEVICE) for idx in [one_idx, zero_idx] ]
        # split d_val into (+) and (-)
        one_idx_dval, zero_idx_dval = [torch.where(dval_data[:, 2] == i)[0] for i in [1, 0]]
        one_data_dval, zero_data_dval = [dval_data[idx].to(args.DEVICE) for idx in [one_idx_dval, zero_idx_dval]]
        # split h_val into (+) and (-)
        one_idx_hval, zero_idx_hval = [torch.where(hval_data[:, 2] == i)[0] for i in [1, 0]]
        one_data_hval, zero_data_hval = [hval_data[idx].to(args.DEVICE) for idx in [one_idx_hval, zero_idx_hval]]
        # split v_val into (+) and (-)
        one_idx_vval, zero_idx_vval = [torch.where(vval_data[:, 2] == i)[0] for i in [1, 0]]
        one_data_vval, zero_data_vval = [vval_data[idx].to(args.DEVICE) for idx in [one_idx_vval, zero_idx_vval]]

        # enzyme embeddings
        file = '/cluster/scratch/ovavourakis/embeds/esm2_3B_hidden.pkl'
        with open(file, 'rb') as f:
            raw_enz_embeds = pickle.load(f)[:, :, :512].to(args.DEVICE)  # nseqs, maxlen, embedding dim # TODO: remember dimension

        # enzyme embedding padding masks (embedding matrices themselves are read from file every epoch)
        file = 'padding_masks.pkl'
        with open(file, 'rb') as f:
            padding_masks = pickle.load(f)  # nseqs, maxlen
        padding_masks.to(args.DEVICE)

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

        # construct model and copy across GPUs
        num_gpus = torch.cuda.device_count()
        model = Recommender(num_cpd=N_CPD, num_enzyme=N_ENZ, hidden_dim=args.DIM, dropout=args.DROPOUT,
                            mf_on=args.MF_on, mlp_on=args.MLP_on, sep_embeds=args.SEP_EMBEDS)#.to(args.DEVICE)

        print('\n---- MODEL STRUCTURE ----')
        print(f'EMBED DIM:\t {args.DIM}')
        print(f'MF_track:\t {model.mf_on}')
        print(f'MLP_track:\t {model.mlp_on}')
        print(f'SEPARATE EMBEDS: {model.sep_embeds}')
        print(f'TOTAL PARAMS:\t {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n')

        if num_gpus > 1:
            model = MyDataParallel(model)
        model = model.to(args.DEVICE)

        opt = torch.optim.AdamW(model.parameters(), lr=args.LR, weight_decay=args.L2)

        print('\n---- TASKS ----')
        print(f'AUTOENC: {args.AUTOENC}')
        print(f'EC: {args.EC}')
        print(f'KO: {args.KO}')
        print(f'CC: {args.CC}')

        # current_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
        # print(f"Current GPU memory usage: {current_memory:.2f} GB")

        train(model, args.OUTF)