import torch, pickle, sys, subprocess
import numpy as np
from tqdm import tqdm

def show_gpu(msg):
    """
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """
    def query(field):
        return (subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
             '--format=csv,nounits,noheader'],
            encoding='utf-8'))

    def to_int(result):
        return int(result.strip().split('\n')[0])

    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used / total
    print('\n' + msg, f'{100 * pct:2.1f}% ({used} out of {total})')

show_gpu('Initial GPU memory usage:')

print('LOADING MODEL')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D") # use 4 CPUs -> 16GB x 4 RAM
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t36_3B_UR50D")     # use 8 CPUs
model = model.to(device)

batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# show_gpu('GPU memory usage after loading model:')

print('LOADING SEQ DATA')
def crop(seq, maxlen=1024):
    length = len(seq)
    if length > maxlen:
        start_index = np.random.randint(0, length - maxlen)
        seq = seq[start_index:start_index + maxlen]
        assert(len(seq)<=maxlen)
    return seq

SEQ_DIR = '/cluster/project/krause/ovavourakis/boost-rs-repl/data/EC_seqs/'
SEQ_FILE = SEQ_DIR+'all_seqs.pkl'
with open(SEQ_FILE, 'rb') as fi:
    all_seqs = pickle.load(fi)
all_seqs = all_seqs['ref_seq'].apply(lambda seq: crop(seq))

assert all_seqs.index.is_monotonic_increasing, "sequences are not ordered by index!"
for i in range(4617):
    assert(i in all_seqs.index)

data = []
for index, seq in enumerate(all_seqs):
    data.append((str(index), seq))

print('TOKENIZING')
data_labels, data_strs, data_tokens = batch_converter(data)  # to pad everything to max_length uniformly
data_lens = (data_tokens != alphabet.padding_idx).sum(1)

batch_size = 10  # small so that it also works for the 3B model
batches = [data_tokens[i:i+batch_size, :] for i in range(0, len(data_tokens), batch_size)]

show_gpu(f'GPU memory usage:')

print('EMBEDDING')
hidden_matrices = []
first_columns = []
means_over_length = []

for batch_tokens in tqdm(batches):
    with torch.no_grad():
        results = model(batch_tokens.to(device), repr_layers=[33], return_contacts=False)
        hidden_matrix = results["representations"][33]  # batchsize, maxlen of batch, dim

        if len(hidden_matrices) == 1 :
            show_gpu(f'GPU memory usage:')

        hidden_matrices.append(hidden_matrix.cpu())  # important: move to CPU (else will accumulate in GPU RAM)
        first_columns.append(hidden_matrix[:, 1, :].cpu())
        means_over_length.append(hidden_matrix[:, 1:-1, :].mean(1).cpu())

hidden_matrices = torch.vstack(hidden_matrices)  # this operation uses a lot of (CPU) memory
first_columns = torch.vstack(first_columns)
means_over_length = torch.vstack(means_over_length)

OUTDIR = '/cluster/project/krause/ovavourakis/boost-rs-repl/data/seq_embeds/'

# FILE = '/cluster/scratch/ovavourakis/embeds/esm2_650M_hidden.pkl'
FILE = '/cluster/scratch/ovavourakis/embeds/esm2_3B_hidden.pkl'
with open(FILE, 'wb') as f:
    pickle.dump(hidden_matrices, f)

# FILE = OUTDIR + 'esm2_650M_first.pkl'
FILE = OUTDIR+'esm2_3B_first.pkl'
with open(FILE, 'wb') as f:
    pickle.dump(first_columns, f)

# FILE = OUTDIR + 'esm2_650M_mean.pkl'
FILE = OUTDIR + 'esm2_3B_mean.pkl'
with open(FILE, 'wb') as f:
    pickle.dump(means_over_length, f)