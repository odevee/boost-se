"""
- searches KEGG for GENES (listed on EC page) and each GENE : UniProtID
    - takes all hits, if >50 AA long
    - finds their most common UniRef90 cluster with ref_seq > 50 AA (second try with UniRef90)
    - takes that ref_seq as representative of the EC
- if unsuccessful searches BRENDA for EC and takes all UniProt IDs on that page
    - among those UniProtIDs, finds their most common UniRef90 cluster, as before
- if unsuccessful, searches for the EC on UniProt
    - among those UniProtIDs, finds their most common UniRef90 cluster, as before
- do the remaining ones by hand

- then read everything back in, eliminate duplicates by picking at random from an ECs uniprot-seqs
    - discorvered again, aither via KEGG, or BRENDA, or EC-search on UniProt
    - some of this is done by hand for pathological cases

-
"""

import pickle, requests, random
import pandas as pd
from tqdm import tqdm
from Bio.KEGG import Enzyme
from io import StringIO

# ----------------------------------------------------------------------------------------------------------------------
# DEFINITIONS
# ----------------------------------------------------------------------------------------------------------------------

# OUTDIR = '/home/ovavourakis/data/seqs'
OUTDIR = '/Users/odysseasvavourakis/Documents/2022-2023/Studium/5. Semester/Thesis Work/src/boost-rs-repl/data/EC_seqs_from_uniref90_redo'
# OUTDIR = '/Users/odysseasvavourakis/Documents/2022-2023/Studium/5. Semester/Thesis Work/src/boost-rs-repl/data/EC_seqs_from_uniref'

def fetch_upid_from_ec(ec_number):
    # get all UniProt entries for this EC with >50 AA
    url=f'https://rest.uniprot.org/uniprotkb/stream?compressed=false&fields=accession%2Cid&format=tsv&query=%28%28ec%3A{ec_number}%29%29%20AND%20%28length%3A%5B50%20TO%20%2A%5D%29%20'
    response = requests.get(url).text
    response = pd.read_csv(StringIO(response), sep='\t')
    upids = list(response['Entry'])
    if len(upids) == 0:
        failed_ECs.append(ec_number)
        return None
    else:
        return upids

def fetch_unirefs_from_uniprots(up_ids, uniref=90):
    base_url='https://rest.uniprot.org/uniref/stream?fields=id%2Clength%2Csequence&format=tsv&query=%28%28'
    main_url="".join([f'uniprot_id%3A{upid}%29%20OR%20%28' for upid in up_ids[:-1]])
    last_url = f'uniprot_id%3A{up_ids[-1]}%29%20'
    if uniref==50:
        tail_url = 'AND%20%28length%3A%5B50%20TO%20%2A%5D%29%29%20AND%20%28identity%3A0.5%29'  # UniRef50
    else:
        tail_url = 'AND%20%28length%3A%5B50%20TO%20%2A%5D%29%29%20AND%20%28identity%3A0.9%29'  # UniRef90
    url = base_url+main_url+last_url+tail_url
    response = requests.get(url).text
    if response == '':
        raise
    else:
        return pd.read_csv(StringIO(response), sep='\t')

def fetch_random_seq_from_uniprots(upids):
    batch_size = 800
    num_batches = (len(upids) + batch_size - 1) // batch_size
    seqs = []

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = (i + 1) * batch_size
        batch = upids[start_index:end_index]

        base_url='https://rest.uniprot.org/uniprotkb/stream?fields=accession%2Clength%2Csequence&format=tsv&query=%28'
        main_url="".join([f'{upid}%20OR%20' for upid in batch[:-1]])
        last_url=f'{batch[-1]}%20'
        tail_url='AND%20%28length%3A%5B50%20TO%20%2A%5D%29%29'
        url = base_url + main_url + last_url + tail_url
        response = requests.get(url).text
        if response == '':
            raise  # TODO
        else:
            df = pd.read_csv(StringIO(response), sep='\t')
            seqs.append(df.iloc[random.choice(df.index)]['Sequence'])
    return random.choice(seqs)

def get_most_frequent_ref_sequence(uniref_df):
    most_freq_id = uniref_df['Cluster ID'].value_counts().idxmax()
    uniref_df.drop_duplicates(subset='Cluster ID', inplace=True)
    uniref_df.set_index('Cluster ID', inplace=True)
    most_freq_seq = uniref_df.loc[most_freq_id]['Reference sequence']
    return most_freq_id, most_freq_seq

def fetch_unirefs_from_upids_batched(upids, uniref=90):
    batch_size = 800
    num_batches = (len(upids) + batch_size - 1) // batch_size
    chunks = []
    with tqdm(total=num_batches, desc="up_id batches", leave=False) as pbar:
        for i in range(num_batches):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size
            batch = upids[start_index:end_index]
            chunk = fetch_unirefs_from_uniprots(batch, uniref=uniref)
            chunk = chunk.drop(chunk[chunk['Length'] < 50].index)
            chunks.append(chunk)
            pbar.update(1)
    chunks = pd.concat(chunks, axis=0)
    if not chunks.empty:
        return get_most_frequent_ref_sequence(chunks)
    else:
        return None, None

def get_refseq_for_ec(ec_number, uniref=90, random=False):
    upids = fetch_upid_from_ec(ec_number)

    if upids:
        if random:
            return fetch_random_seq_from_uniprots(upids)
        return fetch_unirefs_from_upids_batched(upids, uniref=uniref)
    else:
        return None, None

def read_genes(EC_FILE):
    with open(EC_FILE, 'r') as file:
        all_genes = []
        for EC in list(Enzyme.parse(file)):
            genelist = EC.genes
            for gene_set in genelist:
                organism, org_genes = gene_set
                organism = organism.lower()
                all_genes.extend([f'{organism}:{gene}' for gene in org_genes])
    return all_genes

# which ECs do we have
EC_KEY = './data/block_splits/enzyme_key.pkl'
with open(EC_KEY, 'rb') as fi:
    Eid_to_EC = pickle.load(fi)
    EC_to_Eid = {value: key for key, value in Eid_to_EC.items()}
keylist = list(EC_to_Eid.keys())

# get all the KEGG gene ids for each EC
print('GET GENES FOR EACH EC')
EC_DIR = '/Users/odysseasvavourakis/Documents/2022-2023/Studium/5. Semester/Thesis Work/datasets.nosync/kegg/ligand/enzyme/enzymes_single_entries'
EC_to_gene = {}
for EC in tqdm(keylist):
    EC_file = EC_DIR + '/' + EC
    EC_to_gene[EC] = read_genes(EC_file)
all_genes = [string for sublist in EC_to_gene.values() for string in sublist]

# get the UniProt ID of each gene in the database
print('PARSE UNIPROT IDs for EACH GENE')
GENE_UniProt_FILE = '/Users/odysseasvavourakis/Documents/2022-2023/Studium/5. Semester/Thesis Work/datasets.nosync/kegg/genes/links/genes_uniprot.list'
print('--- reading file')
gene_to_up = pd.read_csv(GENE_UniProt_FILE, sep='\t', header=None)
print('--- drop unused ECs')
gene_to_up = gene_to_up[gene_to_up[0].isin(all_genes)]
print('--- drop prefixes')
gene_to_up[1]=gene_to_up[1].str.split(':').apply(lambda list : list[1])  # drop "up:" prefix
print('--- convert to dict')
gene_to_up = gene_to_up.groupby(by=gene_to_up[0]).agg(set).to_dict(orient='dict')[1]

# # ----------------------------------------------------------------------------------------------------------------------
# # EC -(KEGG)-> KEGG Genes -(Uniprot > 50 AA)-> UniRef50/90
# # ----------------------------------------------------------------------------------------------------------------------
#
# subl_size = len(keylist) // 10
# sublists = [keylist[i:i + subl_size] for i in range(0, len(keylist), subl_size)]
#
# failed_ECs = []  # these will fail to get any upids
#
# for s, sublist in enumerate(sublists):
#     print(f'SUBLIST: {s+1}/11')
#     EC_to_refseq = pd.DataFrame()
#     for EC in tqdm(sublist, desc="EC Progress"):
#         upids = []
#         upids_found = False
#         genes = EC_to_gene[EC]  # get KEGG gene ids for this EC (if annotated in KEGG)
#         for gene in genes:
#             up = gene_to_up.get(gene)  # key might not exist
#             if up:
#                 upids.append(list(up)[0])  # get it out of the set
#                 upids_found = True
#         if not upids_found:
#             failed_ECs.append(EC)
#         else:
#             uniref_id, ref_seq = fetch_unirefs_from_upids_batched(upids)
#             if uniref_id and ref_seq:
#                 EC_to_refseq = EC_to_refseq.append(pd.DataFrame([[EC, uniref_id, ref_seq]]), ignore_index=True)
#     EC_to_refseq[0] = EC_to_refseq[0].apply(lambda ec: EC_to_Eid[ec])
#
#     EC_to_refseq = EC_to_refseq[[0,2]]
#     EC_to_refseq.columns = ['Eid', 'ref_seq']
#     EC_to_refseq = EC_to_refseq.set_index('Eid')
#
#     FNAME = f'EC_seqs_from_uniref90_via_KEGGgenes_{s}.pkl'
#     with open(OUTDIR+'/'+FNAME, 'wb') as fi:
#         pickle.dump(EC_to_refseq, fi, protocol=pickle.HIGHEST_PROTOCOL)
#
#     print(f'STAGE 1 FAILED ECs (so far):\n {failed_ECs}')
#
# # ----------------------------------------------------------------------------------------------------------------------
# # EC -(BRENDA)-> UniProt -(Uniprot > 50 AA)-> UniRef50/90
# # ----------------------------------------------------------------------------------------------------------------------
#
# # failed_ECs =  ['1.1.1.144', '1.1.1.207', '1.1.1.223', '1.1.1.247', '1.1.1.257', '1.1.1.266', '1.1.1.309', '1.1.1.339', '1.1.1.356', '1.1.3.38', '1.13.11.45', '1.13.11.89', '1.13.12.13', '1.13.12.24', '1.13.12.7', '1.14.11.20', '1.14.11.49', '1.14.11.70', '1.14.11.71', '1.14.13.101', '1.14.13.146', '1.14.13.16', '1.14.13.189', '1.14.13.210', '1.14.13.222', '1.14.14.102', '1.14.14.105', '1.14.14.106', '1.14.14.124', '1.14.14.125', '1.14.14.141', '1.14.14.145', '1.14.14.168', '1.14.14.169', '1.14.14.170', '1.14.14.183', '1.14.14.41', '1.14.14.60', '1.14.14.66', '1.14.14.85', '1.14.14.97', '1.14.14.99', '1.14.19.10', '1.14.19.66', '1.17.5.2', '1.2.1.62', '1.2.1.73', '1.21.3.3', '1.3.1.36', '1.3.1.81', '1.3.98.4', '1.5.3.19', '1.5.3.6', '1.97.1.1', '2.1.1.102', '2.1.1.115', '2.1.1.118', '2.1.1.122', '2.1.1.158', '2.1.1.159', '2.1.1.262', '2.1.1.263', '2.1.1.284', '2.1.1.300', '2.1.1.330', '2.1.1.47', '2.1.1.94', '2.1.1.99', '2.1.2.8', '2.3.1.107', '2.3.1.146', '2.3.1.150', '2.3.1.160', '2.3.1.162', '2.3.1.166', '2.3.1.197', '2.3.1.206', '2.3.1.216', '2.3.1.236', '2.3.1.238', '2.3.1.239', '2.3.1.240', '2.3.1.290', '2.3.1.307', '2.3.2.19', '2.4.1.116', '2.4.1.219', '2.4.1.238', '2.4.1.299', '2.4.1.301', '2.4.1.327', '2.4.1.330', '2.4.1.362', '2.4.1.363', '2.4.1.366', '2.4.2.55', '2.6.1.90', '2.6.1.94', '2.7.4.12', '2.7.4.13', '2.8.2.25', '2.8.2.27', '3.1.1.78', '3.1.1.80', '3.1.1.91', '3.1.2.31', '3.1.3.33', '3.1.3.34', '3.13.2.2', '3.2.1.117', '3.2.1.125', '3.5.1.123', '3.5.1.91', '3.5.1.93', '3.5.2.12', '3.5.5.6', '3.8.1.7', '4.1.1.95', '4.1.2.47', '4.2.1.106', '4.2.1.145', '4.2.3.132', '4.2.3.198', '4.2.3.204', '4.2.3.24', '4.2.3.31', '4.2.3.32', '4.2.3.50', '4.2.3.53', '4.2.3.54', '4.2.3.56', '4.2.3.58', '4.2.3.74', '4.2.3.92', '4.3.2.6', '4.5.1.1', '4.8.1.1', '5.3.2.3', '5.3.3.13', '5.4.3.10', '5.4.99.46', '5.4.99.51', '5.5.1.15', '6.1.2.2', '6.3.1.18', '6.7.1.1']
#
BRENDA_FILE = '/Users/odysseasvavourakis/Documents/2022-2023/Studium/5. Semester/Thesis Work/datasets.nosync/brenda/brenda_2023_1.txt'
BRENDA_EC_to_UP = {}
found_first_id = False
within_id = False
with open(BRENDA_FILE, 'r') as file:
    line = file.readline()
    while line:
        line = line.strip()

        if not found_first_id and line.startswith("ID"):
            found_first_id = True

        elif found_first_id and line.startswith("ID"):
            list = line.strip().split()
            EC = list[1]
            within_id = True

        while within_id:
            UP_IDs = []

            while not line.startswith('PR'):
                line = file.readline()

            while line.startswith('PR'):
                if "UniProt" in line:
                    words = line.strip().split()
                    index = words.index("UniProt")

                    uniprot_id = words[index - 1]
                    UP_IDs.append(uniprot_id)
                line = file.readline()

            if len(UP_IDs) > 0:
                BRENDA_EC_to_UP[EC] = set(UP_IDs)

            within_id = False
        line = file.readline()

del list
#
# print('RE-TRYING FAILED ECs via BRENDA EC pages -> UniProt')
# still_none_found = []
# BRENDA_EC_to_refseq = pd.DataFrame()
# for EC in tqdm(failed_ECs):
#     upids = BRENDA_EC_to_UP.get(EC)  # key might not exist
#     if upids is None:
#         still_none_found.append(EC)
#     else:
#         uniref_id, ref_seq = fetch_unirefs_from_upids_batched(list(upids))
#         BRENDA_EC_to_refseq = BRENDA_EC_to_refseq.append(pd.DataFrame([[EC, uniref_id, ref_seq]]), ignore_index=True)
# BRENDA_EC_to_refseq[0] = BRENDA_EC_to_refseq[0].apply(lambda ec: EC_to_Eid[ec])
#
# BRENDA_EC_to_refseq = BRENDA_EC_to_refseq[[0,2]]
# BRENDA_EC_to_refseq.columns = ['Eid', 'ref_seq']
# BRENDA_EC_to_refseq = BRENDA_EC_to_refseq.set_index('Eid')
#
# FNAME = f'EC_seqs_from_uniref90_via_BRENDA.pkl'
# with open(OUTDIR + '/' + FNAME, 'wb') as fi:
#     pickle.dump(BRENDA_EC_to_refseq, fi, protocol=pickle.HIGHEST_PROTOCOL)
#
# print(f'STAGE 2 FAILED ECs:\n {still_none_found}')
#
# # ----------------------------------------------------------------------------------------------------------------------
# # EC -(Uniprot > 50 AA)-> UniRef50/90
# # ----------------------------------------------------------------------------------------------------------------------
#
# failed_ECs = []  # these will fail to get any upids
# EC_to_refseq = pd.DataFrame()
#
# for EC in tqdm(still_none_found, desc="EC Progress"):
#     uniref_id, ref_seq = get_refseq_for_ec(EC)  # UniRef90 by default
#     if uniref_id and ref_seq:
#         EC_to_refseq = EC_to_refseq.append(pd.DataFrame([[EC, uniref_id, ref_seq]]), ignore_index=True)
# EC_to_refseq[0] = EC_to_refseq[0].apply(lambda ec: EC_to_Eid[ec])
#
# final_Eid_to_refseq = EC_to_refseq[[0,2]]
# final_Eid_to_refseq.columns = ['Eid', 'ref_seq']
# final_Eid_to_refseq = final_Eid_to_refseq.set_index('Eid')
#
# FNAME = f'EC_seqs_from_uniref90_via_uniprot_directly.pkl'
# with open(OUTDIR+'/'+FNAME, 'wb') as fi:
#     pickle.dump(final_Eid_to_refseq, fi, protocol=pickle.HIGHEST_PROTOCOL)
#
# print(f'STAGE 3 FAILED ECs:\n {failed_ECs}')
# # failed_ECs=['1.13.12.24', '1.14.11.70', '1.14.14.183', '1.14.14.66', '2.3.1.290', '2.3.1.307', '2.4.1.362', '2.4.2.55', '2.7.4.12', '3.13.2.2', '3.5.1.123', '3.5.5.6', '4.8.1.1']
#
# # ----------------------------------------------------------------------------------------------------------------------
# # EC -(manual assignment)-> UniRef50/90 [EC -> KO -> KEGG GENEs -> (AAseq directly OR UniProt)]
# # ----------------------------------------------------------------------------------------------------------------------
#
# data90 = [
#     ('1.13.12.24','UniRef90_P02592','MTSKQYSVKLTSDFDNPRWIGRHKHMFNFLDVNHNGKISLDEMVYKASDIVINNLGATPEQAKRHKDAVEAFFGGAGMKYGVETDWPAYIEGWKKLATDELEKYAKNEPTLIRIWGDALFDIVDKDQNGAITLDEWKAYTKAAGIIQSSEDCEETFRVCDIDESGQLDVDEMTRQHLGFWYTMDPACEKLYGGAVP'),
#     ('1.14.11.70','UniRef90_B0LI31','MTIYENKLSSYQKNQDAIISAKELEEWHLIGLLDHSIDAVIVPNYFLEQECMTISERIKKSKYFSAYPGHPSVSSLGQELYECESELELAKYQEDAPTLIKEMRRLVHPYISPIDRLRVEVDDIWSYGCNLAKLGDKKLFAGIVREFKEDNPGAPHCDVMAWGFLEYYKDKPNIINQIAANVYLKTSASGGEIVLWDEWPTQSEYIAYKTDDPASFGLDSKKIAQPKLEIQPNQGDLILFNSMRIHAVKKIETGVRMTWGCLIGYSGTDKPLVIWT'),
#     ('1.14.14.183','UniRef90_A0A087Y5Y3','METEKRTQKSAEQTVTRDLSEQIKEVTKESHVRAENTELMLSFQRGQVTLPQYKLLLCSLYEIYRALEEEMDRNSNHPAVAPIYFPAELARIKSIEKDLEYFYGPDWREKIVVPVATQRYSHRLRQIGKENPEFLVAHAYTRYLGDLSGGQVLGRIAQKSMGLKSGEGLSFFAFPGVSSPNLFKQLYRSRMNSVELTAEERNGVLEEAVRAFEFNIQVFDGLQKMLSVAGNQQRQSSTDSKAVHGKTLQILGTFSPMLRMVLGLFVALAAVVGLYAL'),
#     ('1.14.14.66', 'KEGG GENE 115714670','MNMEFYTYDMLSDLLYLMLTIFISFQIFNLLSSYTKTQQQLPPGPRPLPIIGNLLELGHNPHKSLARLSQIHGPIIYLKLGQTTTVVVSSAEMAKQILQTHDNLFSNRTVPDSFRAYNHDKHSVSLSPVSPSWRNLRKICHNHLFSVKALDSNQNLRQQKVQELLADVRTMGEASNEAVAIGVAGFKTTLNLLSTTFFSADWENLAASDMAVDLKETIGNVMVAGGKPNLSDYFPVLRKLDLLGLRRSMTFHFRKLKDFFDGMITQRLKHREESNSPKGEEENNNMLDTLLNYMASKENSENQLLNKTTIEHLLLDLFAAGTDTTSSTLEWSMTELLKNPEAMSKAQAELNQVIGKGNQMKESDITRLPYLQAVIKETFRLHPPAPLLLPRRAETDVELCGYVVPEGAQVLVNAWAISRDPNIWDNPNEFIPERFLESDIDVKGRHFELTPFGGGRRICPGLPLAIRMVHLMLGSLIHSFDWKLEDGVKPQTLNMDDKFGLTLHKAQPLKALPIPIIST'),
#     ('2.3.1.290','KEGG GENE CAO85893','MELNGIRRRLATAKEAERHRLLLELIRAAAAEALDRPGPLPLDPGRAFGAQGVRGRAAGRLRERLSEATGLPLPATVVFDYPTPGALAGRLCALLLDEPAGPGGANGQMEAEEPIAIVGMGCRLPGDVRSPEGLWRLVHNGTDAISEFPEDRGWTVQQHPDPDHLGTTVTRHAGFLYDAPDFDAGFFAISPGEAVTIDPQHRLLLETTWKAVEDARIDPTSLRGSRTGVFVGLMYSEYGARIRRVPPGAEGYRVVGSMPSVASGRLAYTFGFEGPAVTVDTACSSSLVAMHLAAQSLRKGECTLAVAGGATVMATPWGYIEFSRQRGLAPDGRCRSFSADAAGSSWSEGVGVLLLERLSDARRHGHRVLAVVRGSAVNQDGASNGLTAPNGPAQQRVIRQALAHAGLTTAEVDAVDAHGAGTRLGDPIEAQALLATYGQGRPAGRPLWLGSLKSNIGHTQAAAGAAGVIKMVMAMRHGVLPRSLHITEPTPHVDWTSGAVELLTEARDWPADGRPRRAAVSSFGVGGTNAHIILEQAAPEPERPHAPEADGEPRPLPWPVSGHGAAGLRAQARRLADFLRAGPAAPDADLAYSLATTRATLTDRAVVVAADRAEAIARLTALAEGDQGPRVARATAVPRDRLAFVFPGQGSQWPGMAAELMSCYPVFRESIKECGRSLAPHTDWSLAKVLRGESGAPTLDRVDVVQPALFAVMVSLAALWRSFGVEPSAVAGHSQGEIAAARVAGALSLEDAARVVALRSRALRVLSGRGGMVSVAAPAGQVLRTLERWGGAVSVAAVNGPRSLVISGDPGALGEALAAFEAEGIRARRIPVDYASHSAQVEEIRDTLLTELSGIRPRPATVPFYSTVSGEPLDTTALDTGYWVRNLRDTVQFDRTVRRLLADGHTTFLEMSPHPVLTPGIQETAEEAGADEVLTVESLRRNEGGPARLLTAVAEAHVHGVAVDWSPAFAPHRSPARRPAVLRLSSGAATGLRTRPRPPPMFTTAGLDGIDHPLLGAAIPLADGGGGTLFTGTLSLATHPWLADHAVADVLVVPGTALVEAALRAGAECTGRAMLEELVLQAPLILPEQGTVRIQLSVGGPDGTGRRALILSSRPEDAGADEPWTRHAEGTLAPGGGHPRQDPGPWPPTGAREIDLDDCYRQLAKTGLHYGPAFQGLKRLWRLADDLCLEAELPDSAGESGRYGLHPALFDAALHAAALAGPSGAEPLTRLPFSWSGVALYTAGATRLRARLSFTGPQSLTLTAMDPLGHPVLSVGTLGMRPVTAEALHRAAGTAGTALLRLEWRRQAPEHPAGPDLTGWAWVGAGAPPAQPPDGRPYRDLAALRAELDAGAAVPPVIVLAEPATPEGTDPFTAARAALHRTLAAIQDWAAEERLAGTRLVVLTQGAVAATPGALPDPALAAVWGLVRSAQAEYPDRIGLVDTDDPGRSRAAVAAAVHAGEAQAAVRDGALLVPRLARVTATDGPGGPAWPADGTVLVTGGLGTLGRLVVRHLVTTHGARRLVILSRSGGDSAEVREFVGELLAQGANVQVVKGDAADPAVLERVLDGIPQEHPLAAVAHLAGALDDGVLAAQTPQRLDRVLRPKAEAAWQLHRLTARARVPLLAFSSLSGVLGPAGQAGYAAANAFVDALVQRRRGTGLPGVSMGWGMWATRSGLTGALSDTDARLIARTGVRPLTDEEGLALFDQARATGEPVVFPLGLDITALNSGAPDGIPPLLRGLTARTPARRAGAAEAPEPAGEDLAARLAATPEAERDALLLGVVRGHIAAVLGYDDPRAVAERRPFSDIGFDSLRALQLRNRLGAATGRRLPATLVFDHPNPAALSRYLRTLLLPDPAPAPTAPDGQPGPDQADQVIERLNSASLEEVLDFIDHQLGE'),
#     ('2.3.1.307','KEGG BCN13445','MTTTQQQPVLPAVADILGSAAEGCETLDLGAGLTATVIVRPGAVLNHSAQERVYAALIPVTTSSFGADMTPYWAQRSKEGYLERLAEFVLIADEDGRMVGWTGFHVLPYDGFTLVYLDSTGMVPTRQSGGVMRQLMQARVSGSVAGCPAGKPVYLTARTESPIFYRLMRRLLPAPDALYPQPATAAPGDVVEAARHLARWLGQQDILDAPALTVRGAYDALDELYGELPTTGDPELDKLFRGQLGPLDAFLLVGRVR'),
#     ('2.4.1.362','KEGG CDX65123','MEMKETITRKKLYKSGKSWVAAATAFAVMGVSAVTTVSADTQTPVGTTQSQQDLTGQTGQDKPTTKEVIDKKEPVPQVSAQNVGDLSADAKTPKADDKQDTQPTNAQLPDQGNKQTNSNSDKGVKESTTAPVKTTDVPSKSVAPETNTSINGGQYVEKDGQFVYIDQSGKQVSGLQNIEGHTQYFDPKTGYQTKGELKNIDDNAYYFDKNSGNGRTFTKISNGSYSEKDGMWQYVDSHDKQPVKGLYDVEGNLQYFDLSTGNQAKHQIRSVDGVTYYFDADSGNATAFKAVTNGRYAEQTTKDKDGNETSYWAYLDNQGNAIKGLNDVNGEIQYFDEHTGEQLKGHTATVDGTTYYFEGNKGNLVSVVNTAPTGQYKINGDNVYYLDNNNEAIKGLYGINGNLNYFDLATGIQLKGQAKNIDGIGYYFDQNNGNGEYRYSLTGPVVKDVYSQHNAVNNLSANNFKNLVDGFLTAETWYRPAQILSHGTDWVASTDKDFRPLITVWWPNKDIQVNYLKLMQQIGILDNSVVFDTNNDQLVLNKGAESAQIGIEKKVSETGNTDWLNELLFAPNGNQPSFIKQQYLWNVDSEYPGGWFQGGYLAYQNSDLTPYANTNPDYRTHNGLEFLLANDVDNSNPVVQAEQLNWLYYLMNFGQITANDSNANFDSMRIDAISFVDPQIAKKAYDLLDKMYGLTDNEAVANQHISIVEAPKGETPITVEKQSALVESNWRDRMKQSLSKNATLDKLDPDPAINSLEKLVADDLVNRSQSSDKDSSTIPNYSIVHAHDKDIQDTVIHIMKIVNNNPNISMSDFTMQQLQNGLKAFYEDQHQSVKKYNQYNIPSAYALLLTNKDTVPRVFYGDMYQDYGDDLDGGQYMATKSIYYNAIEQMMKARLKYVAGGQIMAVTKIKNDGINKDGTNKSGEVLTSVRFGKDIMDAQGQGTAESRNQGIGVIVSNSSGLELKNSDSITLHMGIAHKNQAYRALMLTNDKGIVNYDQDNNAPIAWTNDHGDLIFTNQMINGQSDTAVKGYLNPEVAGYLAVWVPVGANDNQDARTVTTNQKNTDGKVLHTNAALDSKLMYEGFSNFQKMPTRGNQYANVVITKNIDLFKSWGITDFELAPQYRSSDGKDITDRFLDSIVQNGYGLSDRYDLGFKTPTKYGTDQDLRKAIERLHQAGMSVMADFVANQIYGLHADKEVVSAQHVNINGDTKLVVDPRYGTQMTVVNSVGGGDYQAKYGGEYLDTISKLYPGLLLDSNGQKIDLSTKIKEWSAKYLNGSNIPQVGMGYVLKDWNNGQYFHILDKEGQYSLPTQLVSNDPETQIGESVNYKYFIGNSDATYNMYHNLPNTVSLINSQEGQIKTQQSGVTSDYEGQQVQVTRQYTDSKGVSWNLITFAGGDLQGQKLWVDSRALTMTPFKTMNQISFISYANRNDGLFLNAPYQVKGYQLAGMSNQYKGQQVTIAGVANVSGKDWSLISFNGTQYWIDSQALNTNFTHDMNQKVFVNTTSNLDGLFLNAPYRQPGYKLAGLAKNYNNQTVTVSQQYFDDQGTVWSQVVLGGQTVWVDNHALAQMQVSDTSQQLYVNSNGRNDGLFLNAPYRGQGSQLIGMTADYNGQHVQVTKQGQDAYGAQWRLITLNNQQVWVDSRALSTTIVQAMNDDMYVNSNQRTDGLWLNAPYTMSGAKWAGDTRLANGRYVHISKAYSNEVGNTYYLTNLNGQSTWIDKRAFTATFDQVVALNATIVARQRPDGMFKTAPYGEAGAQFVDYVTNYNQQTVPVTKQHSDAQGNQWYLATVNGTQYWIDQRSFSPVVTKAVDYQAKIVPRTTRDGVFSGAPYGEVNAKLVNMATAYQNQVVHATGEYTNASGITWSQFALSGQEDKLWIDKRALQA'),
#     ('2.4.2.55','UniRef90_A0A0U1KTQ7','MNWEPGDLPYTATILHGQEGFFGLNQKSLLFLSRWKVDKNKRLLLSILVMEVIMSLLQATVAKIMRPDTVIKDQVKTKLAGVLQSAGSLGRLEDMVEQYAGITGELNPALPKPCMVVASADHGVARRVVSAYPIETTIHMTANYLISQGASANAFANFCGADMVVVDMGVAGDLSYVPGLWHRKIAYGTQDFTEGPAMTREQAIQAVETGIDIVNDRVKHGNRCFCLGEMGIGNTTSSATIVGAFTGLAPEKVTGRGTGISDSRLKTKMEIVGRALAVNKPNPQDGLDVLAKVGGFELGALAGVILGSAANRCAVVIDGLNTTAAALIANVIHPLSKEYMFASHLSGEPAHSIALRQLQLEACLELGVRLGEGIGASMVVDMLYVAIKLLNNRGGKANA'),
#     ('2.7.4.12','UniRef90_P04531','MKLIFLSGVKRSGKDTTADFIMSNYSAVKYQLAGPIKDALAYAWGVFAANTDYPCLTRKEFEGIDYDRETNLNLTKLEVITIMEQAFCYLNGKSPIKGVFVFDDEGKESVNFVAFNKITDVINNIEDQWSVRRLMQALGTDLIVNNFDRMYWVKLFALDYLDKFNSGYDYYIVPDTRQDHEMDAARAMGATVIHVVRPGQKSNDTHITEAGLPIRDGDLVITNDGSLEELFSKIKNTLKVL'),
#     ('3.13.2.2','UniRef90_P07693','MIFTKEPAHVFYVLVSAFRSNLCDEVNMSRHRHMVSTLRAAPGLYGSVESTDLTGCYREAISSAPTEEKTVRVRCKDKAQALNVARLACNEWEQDCVLVYKSQTHTAGLVYAKGIDGYKAERLPGSFQEVPKGAPLQGCFTIDEFGRRWQVQ'),
#     ('3.5.1.123','UniRef90_A0A2J7Y1Y2','MSKRFALLWCSEEERFDYREEMVNAFKTENSDWEVISAFTDLNKIIDNYDGFVISGSEYSVNADKEKFSGLFEFIRAVHKKEKPIVGICFGCQSLAVALGGEVGLNPSREFRFGTDELTFQNGLNKHVGTSEERVRLIESHGECVIRRPLGSTLLARSDSTAVEIFAVGPYAVGIQGHPEISKKTLEQVFLRVHLEDGNLQEDEVPRFHAELSGYQPPQAIRQLVKATLHKQINFQNLVGDV'),
#     ('3.5.5.6','UniRef90_P1004','MDTTFKAAAVQAEPVWMDAAATADKTVTLVAKAAAAGAQLVAFPELWIPGYPGFMLTHNQTETLPFIIKYRKQAIAADGPEIEKIRCAAQEHNIALSFGYSERAGRTLYMSQMLIDADGITKIRRRKLKPTRFERELFGEGDGSDLQVAQTSVGRVGALNCAENLQSLNKFALAAEGEQIHISAWPFTLGSPVLVGDSIGAINQVYAAETGTFVLMSTQVVGPTGIAAFEIEDRYNPNQYLGGGYARIYGPDMQLKSKSLSPTEEGIVYAEIDLSMLEAAKYSLDPTGHYSRPDVFSVSINRQRQPAVSEVIDSNGDEDPRAACEPDEGDREVVISTAIGVLPRYCGHS'),
#     ('4.8.1.1','KEGG GENE ABV56600','MFVPGPYHAPEDRWLVDLVRGHPLAQLASNGAGGAAPHITHVPIIVDPELDGPVDRLVGITLWGHMNRANPHWAALGGAANVVATFAGPNAYVSPAVYRTAPAAPTWNFTSVQVRGELRKVESADDTLATVRATVAALESRFGAGWDMTGSLDYFRRILPGVGAFRLRVAEADGMFKLSQEQQPAIRRRVRHSFGGCEATRAVAGLMDRLPTE')
# ]
#
# # data50 = [
# #     ('1.13.12.24','UniRef50_P02592','MTSKQYSVKLTSDFDNPRWIGRHKHMFNFLDVNHNGKISLDEMVYKASDIVINNLGATPEQAKRHKDAVEAFFGGAGMKYGVETDWPAYIEGWKKLATDELEKYAKNEPTLIRIWGDALFDIVDKDQNGAITLDEWKAYTKAAGIIQSSEDCEETFRVCDIDESGQLDVDEMTRQHLGFWYTMDPACEKLYGGAVP'),
# #     ('1.14.11.70','UniRef50_D7F1B0','MTVYENKLGSYKTNQDALLSAKNLEEWHLIGLLDHSIDAVIVPNYFLDQECAIIAERIKKSKYFSAYPDHPSVSRLGQELFECGESELALEKQQEKAPTLMKEMRRLIHPYISPIDRLRVELDDIWSYGCHLAKLGDKKVFAGIVREHKEDSPGVPHCDVMGWGFLKSYKDKPNLINNIAANVYLKMSESGGEIVLWDEWPTQSEYKIERNIDDPASVGDSKKIGQPKLEIRPNQGDLILFNSMRIHAVKKIKTGVRMTWGCLIGYSGTDDPLVIWT'),
# #     ('1.14.14.183','UniRef50_Q6JTJ0','MDALSLVNSTVAKFNEVTQLQASPAILSTALTAIAGIIVLLVITSKRRSSLKLPPGKLGLPFIGETLEFVKALRSDTLRQFVEEREGKFGRVFKTSLLGKPTVILCGPAGNRLVLSNEEKLLHVSWSAQIARILGLNSVAVKRGDDHRVLRVALAGFLGSAGLQLYIGKMSALIRNHINEKWKGKDEVNVLSLVRDLVMDNSAILFFNIYDKERKQQLHEILKIILASHFGIPLNIPGFLYRKALKGSLKRKKILSALLEKRKDELRSRLASSNQDLLSVLLSFRDERGKPLSDEAVLDNCFAMLDASYDTTTSQMTLILKMLSSNPECFEKVVQEQLEIASNKKEGEEITMKDIKAMKYTWQVLQESLRMLSPVFGTLRKTMNDINHDGYTIPKGWQVVWTTYSTHQKDIYFKQPDKFMPSRFEEEDGHLDAYTFVPFGGGRRTCPGWEYAKVEILLFLHHFVKAFSGYTPTDPHERICGYPVPLVPVKGFPIKLIARS'),
# #     ('1.14.14.66', 'UniRef50_A0A2D1PE15','MAXLMSLKLASLTTIVVSSPSMAQQVLQKHDHIFSTRWVPDAVRALHHHQFSVVFMPPNLTWRTIRKICNSHAFSNKSLDASQHLRQKKVEELISYIRGCCEDGVAVDIGQTVFSSNLNITSNTFFSMDLAELDSAFNHDFKQIIWKMAHEAGKPNLGDYFPVIRGLDLQGIRHRMEEYHGILIXLFQSIIDQRLANKGVDTSAEISDVLDAFLKVSQEDVDKLDMSYIPHSLVDFFIGGTETASSTIEWAMAELLRDQSKLKKAQAELKATIGKGNTVKEADITRLPYLQAIVKETLRLHPPAPLLVPREANLDVKFCGFKVPKGSQVLINAWAIGRDTSVWERPSCFEPERFLGSDVDVRGLHFELTPFGGGRRICPGINLGVRVLHLIIGSLIHSFDWKLEDGVEPENLDMEEKFGMAVQKAKPLLAIPVISSRC'),
# #     ('2.3.1.290','UniRef50_A0A5B7V304','MTEEATVERNEGRDRALRRRLADAGDVERHRLLLDLIRTHAAAALERTDPAPLDAARAFGAQGVRGRAAARLRERLGEATGVALSATALFDYPTPEALAAHLCSRVLGEPATGDADATASGAAVDEPIAIVGMGCRLPGEVRSPDDLWHLVRSGTDAISEFPRDRGWTVAHDPDPDHVGTTVTRHAGFLYDAADFDADFFGISPGEAVTIDPQHRLLMETAWEAVERARIDPTSLRGSSTGVFVGIMYSEYGARIRHVPPSAEGYRVVGSMPSVASGRLAYALGLEGPAVTVDTACSSSLVSLHLAAQSLRKGECRLALAGGVTVMATPWGYVEFSRQRGLAPDGRCRSFSADAAGSSWSEGVGVLLLERLSDAVRNGHRVLAVVRGSAVNQDGASNGLTAPNGPAQQRVIRQALAHAGLTTADVDCVDAHGAGTQLGDPIEAQALLATYGRGRPAEQPLWLGSLKSNIGHTQAAAGVAGVIKMVMAIRHGVLPRSLHISEPTPHVDWSSGAVRLLAGAEPWPATGRPRRAGVSSFGVGGTNAHVIVEEADPEPDGTAPRTTDGADTPGRPLPWLVSGNGAPGLRAQARRLAEFLAPGPDAPDADLAYSLATTRAALADRAVVVAADRAEARTRLAALADATDATGGAYGGPGESTALGTATARDRVTFVFPGQGSQWPGMAAELMSTSPVFRASIEACAASLAPHVDWSLTKVLRGESGAPTLERVDVVQPALFAVMVSLAALWRSYGVEPAAVIGHSQGEIAAAHVVGALSLDDAARVVAVRSRALRVLSGRGGMVSVVAPVDQVRRGLAPWSGAVSVAAVNGPRSVVISGDPTALDAAMAAFERDGVRVRRIPVDYASHSAQVEEIRDTLVEALSGIRPRPAAVPFHSTVTGEPLDTTALDATYWVNNLRDTVQFHRTVTRLLAAGHTTYVEMSPHPVLTIGIQETADEAGAHDALTVESLRRNEGGPARFLTAVAEAHAHGVAVDWTEAFRHAAPRPTDVPTYAFQRRRYWLEDAAPPVADVAAAGLDGVDHPLLGAALPLADGSGGTLFTGSVSARTHPWLADHAVADVLVVPGTALVEAALHAGAQTGCDTLEELVLEAPLILPEQGVVRVQVSVAGPDAAGRRAVTLYSRAQDAGADEPWTRHAGGTLTRAARPAAAPPAAWPPAGAEEAELDDCYRELAKAGLHYGAALQGLRRCWRLGDELYVEAELPEAAGGTDRYGLHPALFDAALHAAALPGGPGAGGGDATTRLPFSWTGVTLHATGAARLRARVRRTGPDALSLTATDPAGRPVVSVESLGMRPVTAEALRRAAALSDASLLRLEWRPEPLPAPRGPGAARWAWIGPYDAAADDALAARGITRTAHPDVAALSAGLAAGDAPPDAVVLAEPATGPGTDLAAAAHRAVHRALAAVRDWAADDRLAHTRLTVLTRDAVAAAPEASPDPALAAVWGLVRSAQAEYPDRISLLDLDGTAASLAALPAAVDSAEAQLAVRAGTPLTPHLARVPAPAGPGDAPAWRTDGTVLVTGGLGTLGRILVRHLVVTHGVRRLTVVARSGTAGAAEFVAELRAAGAELHVVAGDAGDRDVLAGALAALPDDAPLTAVVHLAGVLDDGVLAAQTPERVDRVLRPKADAAWHLHELTAGSGVPLVAFSSVSSALGPAGQAGYAAANAFVDALAARRHAGGAPGLALGWGLWGERSGLTGDLADTDVRRIARSGVKPLSSDQGLALFDLARATGEPVVFPLRLDPAGLRGDTADEVSPLLRGLARTPVRRAAADAGAALGEDSLADRLGRLAAPEQHELLLDLVRGHVAAVLGYDDPRTVGERRPFTDIGFDSLRALQLRNRLGGATGLRLTATLVFDYPTPLALGRHLRTLLVPDADDGPETAPRPAAGSERPGADEAVLAQLGSASREEVFDLIDDMLAE'),
# #     ('2.3.1.307','KEGG BCN13445','MTTTQQQPVLPAVADILGSAAEGCETLDLGAGLTATVIVRPGAVLNHSAQERVYAALIPVTTSSFGADMTPYWAQRSKEGYLERLAEFVLIADEDGRMVGWTGFHVLPYDGFTLVYLDSTGMVPTRQSGGVMRQLMQARVSGSVAGCPAGKPVYLTARTESPIFYRLMRRLLPAPDALYPQPATAAPGDVVEAARHLARWLGQQDILDAPALTVRGAYDALDELYGELPTTGDPELDKLFRGQLGPLDAFLLVGRVR'),
# #     ('2.4.1.362','KEGG CDX65123','MEMKETITRKKLYKSGKSWVAAATAFAVMGVSAVTTVSADTQTPVGTTQSQQDLTGQTGQDKPTTKEVIDKKEPVPQVSAQNVGDLSADAKTPKADDKQDTQPTNAQLPDQGNKQTNSNSDKGVKESTTAPVKTTDVPSKSVAPETNTSINGGQYVEKDGQFVYIDQSGKQVSGLQNIEGHTQYFDPKTGYQTKGELKNIDDNAYYFDKNSGNGRTFTKISNGSYSEKDGMWQYVDSHDKQPVKGLYDVEGNLQYFDLSTGNQAKHQIRSVDGVTYYFDADSGNATAFKAVTNGRYAEQTTKDKDGNETSYWAYLDNQGNAIKGLNDVNGEIQYFDEHTGEQLKGHTATVDGTTYYFEGNKGNLVSVVNTAPTGQYKINGDNVYYLDNNNEAIKGLYGINGNLNYFDLATGIQLKGQAKNIDGIGYYFDQNNGNGEYRYSLTGPVVKDVYSQHNAVNNLSANNFKNLVDGFLTAETWYRPAQILSHGTDWVASTDKDFRPLITVWWPNKDIQVNYLKLMQQIGILDNSVVFDTNNDQLVLNKGAESAQIGIEKKVSETGNTDWLNELLFAPNGNQPSFIKQQYLWNVDSEYPGGWFQGGYLAYQNSDLTPYANTNPDYRTHNGLEFLLANDVDNSNPVVQAEQLNWLYYLMNFGQITANDSNANFDSMRIDAISFVDPQIAKKAYDLLDKMYGLTDNEAVANQHISIVEAPKGETPITVEKQSALVESNWRDRMKQSLSKNATLDKLDPDPAINSLEKLVADDLVNRSQSSDKDSSTIPNYSIVHAHDKDIQDTVIHIMKIVNNNPNISMSDFTMQQLQNGLKAFYEDQHQSVKKYNQYNIPSAYALLLTNKDTVPRVFYGDMYQDYGDDLDGGQYMATKSIYYNAIEQMMKARLKYVAGGQIMAVTKIKNDGINKDGTNKSGEVLTSVRFGKDIMDAQGQGTAESRNQGIGVIVSNSSGLELKNSDSITLHMGIAHKNQAYRALMLTNDKGIVNYDQDNNAPIAWTNDHGDLIFTNQMINGQSDTAVKGYLNPEVAGYLAVWVPVGANDNQDARTVTTNQKNTDGKVLHTNAALDSKLMYEGFSNFQKMPTRGNQYANVVITKNIDLFKSWGITDFELAPQYRSSDGKDITDRFLDSIVQNGYGLSDRYDLGFKTPTKYGTDQDLRKAIERLHQAGMSVMADFVANQIYGLHADKEVVSAQHVNINGDTKLVVDPRYGTQMTVVNSVGGGDYQAKYGGEYLDTISKLYPGLLLDSNGQKIDLSTKIKEWSAKYLNGSNIPQVGMGYVLKDWNNGQYFHILDKEGQYSLPTQLVSNDPETQIGESVNYKYFIGNSDATYNMYHNLPNTVSLINSQEGQIKTQQSGVTSDYEGQQVQVTRQYTDSKGVSWNLITFAGGDLQGQKLWVDSRALTMTPFKTMNQISFISYANRNDGLFLNAPYQVKGYQLAGMSNQYKGQQVTIAGVANVSGKDWSLISFNGTQYWIDSQALNTNFTHDMNQKVFVNTTSNLDGLFLNAPYRQPGYKLAGLAKNYNNQTVTVSQQYFDDQGTVWSQVVLGGQTVWVDNHALAQMQVSDTSQQLYVNSNGRNDGLFLNAPYRGQGSQLIGMTADYNGQHVQVTKQGQDAYGAQWRLITLNNQQVWVDSRALSTTIVQAMNDDMYVNSNQRTDGLWLNAPYTMSGAKWAGDTRLANGRYVHISKAYSNEVGNTYYLTNLNGQSTWIDKRAFTATFDQVVALNATIVARQRPDGMFKTAPYGEAGAQFVDYVTNYNQQTVPVTKQHSDAQGNQWYLATVNGTQYWIDQRSFSPVVTKAVDYQAKIVPRTTRDGVFSGAPYGEVNAKLVNMATAYQNQVVHATGEYTNASGITWSQFALSGQEDKLWIDKRALQA'),
#
# #     ('2.4.2.55','UniRef50_A0A075K6W9','MIQEVISSIKPLDIAAMDKCQVRLDNLTKPLGSLHALEHLARQLAGITRNPRPRDLKKSILVLAADHGVAAENISAYPQEVTVKMIHQIASGGAAINTFAEHVAADLLLIDMGVATDLPSIPALRNEKITYGTSNINQGPAMSREQAIKAIEAGIKIAVEEIKKGVTAIGLGEVGIANTTSGTAIVATYSQIPVSTLTGRGIGITDTMFNKKIKIIENAIAMNRPDLSDPIDVLSKLGGFEIAGLVGVILGGAAGGAALILDGLVTSAAALIAVKLAPQVKDYLIGSHVSPEPAHKFALDLINLPAHLHLNMSLGEGTGAALGMSVLKASLHVLNDMKTFGDAEVAVAQDGPGALKQDKN'),
# #     ('2.7.4.12','KEGG Virus 70896255','MKLIFLSGVKRSGKDTTADFIMNNYSAVKYQLAGPIKDALAYAWGVFAANTDYPCLTRREFEGIDYDRETNLNLTKLEVIMIMEQAFCYLNGKSPIKGVFVFDDEGKESVNFVAFNKITDVINNIEDQWSVRRLMQALGTDLIVNNFDRMYWVKLFSLDYLDKFNSGYDYYIVPDTRQDHEMDAARAMGATVIHVVRPGQKSNDTHITEAGLPIRDGDLVITNDGSLEELFSKIKNTLKVL'),
# #     ('3.13.2.2','UniRef50_P07693','MIFTKEPAHVFYVLVSAFRSNLCDEVNMSRHRHMVSTLRAAPGLYGSVESTDLTGCYREAISSAPTEEKTVRVRCKDKAQALNVARLACNEWEQDCVLVYKSQTHTAGLVYAKGIDGYKAERLPGSFQEVPKGAPLQGCFTIDEFGRRWQV'),
# #     ('3.5.1.123','UniRef50_A0A2T4E077','MKKRFALVWCSEEERFDYKEEMLKAFEMPNSDWHLVSAFSEMEDVIDRYDGFVISGSEYSVNDDAERFLNLFTLIRKAIVKEKPIVGICFGCQSLAIALGGHVGRNLNKKFRFGVDRLRFKDSLRKVVGIVEQDLGLIESHGECVINCPPGSEILAESKSTSIEIFTPGPYALGIQGHPELSKETVETDFLRVHLADRNIQESELEHFRAEILGYKPPELIRKLVKMVLQ'),
# #     ('3.5.5.6','UniRef50_P10045','MDTTFKAAAVQAEPVWMDAAATADKTVTLVAKAAAAGAQLVAFPELWIPGYPGFMLTHNQTETLPFIIKYRKQAIAADGPEIEKIRCAAQEHNIALSFGYSERAGRTLYMSQMLIDADGITKIRRRKLKPTRFERELFGEGDGSDLQVAQTSVGRVGALNCAENLQSLNKFALAAEGEQIHISAWPFTLGSPVLVGDSIGAINQVYAAETGTFVLMSTQVVGPTGIAAFEIEDRYNPNQYLGGGYARIYGPDMQLKSKSLSPTEEGIVYAEIDLSMLEAAKYSLDPTGHYSRPDVFSVSINRQRQPAVSEVIDSNGDEDPRAACEPDEGDREVVISTAIGVLPRYCGHS'),
# #     ('4.8.1.1','UniRef50_A0A5B7V5A7','MFVPRVYREPEESWKIDLVRGNPLGQLVSNGAEGEAPWVTHVPIIIDPRVTEPVTSLSGTTLWGHMNIGNPHWRALGPATPVAVTFSGPHAYVSPTVYETRPAAPTWNFTAVHIAGVLRKVDSTDETLATVQETVRAYEREFGADWSMTESIEYFRRILPGVGAFRIAISLADGMFKLSQEQPPHVRERVRASFACEASTAHREVAALMGRLGTEDRETVSARPAAAPPPSMGTP')
# # ]
#
# by_hand = pd.DataFrame(data90, columns=['Eid', 'UniRef', 'ref_seq'])[['Eid', 'ref_seq']]
# by_hand['Eid'] = by_hand['Eid'].apply(lambda ec: EC_to_Eid[ec])
# by_hand = by_hand.set_index('Eid')
#
# FNAME = f'by_hand_Uniref90.pkl'
# with open(OUTDIR + '/' + FNAME, 'wb') as fi:
#     pickle.dump(by_hand, fi, protocol=pickle.HIGHEST_PROTOCOL)

# ----------------------------------------------------------------------------------------------------------------------
# DEDUPLICATION
# ----------------------------------------------------------------------------------------------------------------------
# all_files = ['EC_seqs_from_uniref90_via_BRENDA.pkl', 'EC_seqs_from_uniref90_via_KEGGgenes_0.pkl',
#              'EC_seqs_from_uniref90_via_KEGGgenes_1.pkl', 'EC_seqs_from_uniref90_via_KEGGgenes_10.pkl',
#              'EC_seqs_from_uniref90_via_KEGGgenes_2.pkl', 'EC_seqs_from_uniref90_via_KEGGgenes_3.pkl',
#              'EC_seqs_from_uniref90_via_KEGGgenes_4.pkl', 'EC_seqs_from_uniref90_via_KEGGgenes_5.pkl',
#              'EC_seqs_from_uniref90_via_KEGGgenes_6.pkl', 'EC_seqs_from_uniref90_via_KEGGgenes_7.pkl',
#              'EC_seqs_from_uniref90_via_KEGGgenes_8.pkl', 'EC_seqs_from_uniref90_via_KEGGgenes_9.pkl',
#              'EC_seqs_from_uniref90_via_uniprot_directly.pkl', 'by_hand_Uniref90.pkl']
#
# all_seqs = pd.DataFrame()
# for file in sorted(all_files):
#     with open(OUTDIR + '/' + file, 'rb') as fi:
#         all_seqs = all_seqs.append(pickle.load(fi))
# dup_seqs = all_seqs[all_seqs.duplicated('ref_seq', keep=False)]  # 474 duplicates
#
#
# keylist = [Eid_to_EC[Eid] for Eid in dup_seqs.index]
# failed_ECs = []
# EC_to_refseq = pd.DataFrame()
#
# # get all the KEGG gene ids for each EC
# print('GET GENES FOR EACH FAILED EC')
# EC_DIR = '/Users/odysseasvavourakis/Documents/2022-2023/Studium/5. Semester/Thesis Work/datasets.nosync/kegg/ligand/enzyme/enzymes_single_entries'
# EC_to_gene = {}
# for EC in tqdm(keylist):
#     EC_file = EC_DIR + '/' + EC
#     EC_to_gene[EC] = read_genes(EC_file)
# all_genes = [string for sublist in EC_to_gene.values() for string in sublist]
#
# # get the UniProt ID of each gene in the database
# print('PARSE UNIPROT IDs for EACH GENE')
# GENE_UniProt_FILE = '/Users/odysseasvavourakis/Documents/2022-2023/Studium/5. Semester/Thesis Work/datasets.nosync/kegg/genes/links/genes_uniprot.list'
# print('--- reading file')
# gene_to_up = pd.read_csv(GENE_UniProt_FILE, sep='\t', header=None)
# print('--- drop unused ECs')
# gene_to_up = gene_to_up[gene_to_up[0].isin(all_genes)]
# print('--- drop prefixes')
# gene_to_up[1]=gene_to_up[1].str.split(':').apply(lambda list : list[1])  # drop "up:" prefix
# print('--- convert to dict')
# gene_to_up = gene_to_up.groupby(by=gene_to_up[0]).agg(set).to_dict(orient='dict')[1]
#
# # get a random uniprotID:seq for each of the failed ECs
# for EC in tqdm(keylist):
#     upids = []
#     upids_found = False
#     genes = EC_to_gene[EC]             # get KEGG gene ids for this EC (if annotated in KEGG)
#     for gene in tqdm(genes):
#         up = gene_to_up.get(gene)      # key might not exist
#         if up:
#             upids.append(list(up)[0])  # get it out of the set
#             upids_found = True
#     if not upids_found:
#         failed_ECs.append(EC)
#     else:
#         ref_seq = fetch_random_seq_from_uniprots(upids)
#         EC_to_refseq = EC_to_refseq.append(pd.DataFrame([[EC, ref_seq]]), ignore_index=True)
#
# FNAME = f'deduplicated_checkpoint1.pkl'
# with open(OUTDIR + '/' + FNAME, 'wb') as fi:
#     pickle.dump(EC_to_refseq, fi, protocol=pickle.HIGHEST_PROTOCOL)
#
# print(failed_ECs)
# # failed_ECs = ['1.1.1.419', '1.1.3.42', '1.1.3.45', '1.14.14.38', '1.14.14.39', '1.3.3.14', '4.2.3.119', '4.2.3.121', '4.2.3.148', '4.2.3.149', '4.2.3.150', '4.2.3.151', '4.2.3.174', '4.2.3.34', '4.2.3.68', '4.2.3.71', '4.2.3.76', '4.2.3.84', '4.2.3.85', '4.2.99.22', '4.2.99.23', '5.5.1.20', '5.5.1.34', '1.13.11.45', '1.14.14.124', '1.14.14.125', '1.14.14.168', '1.14.14.170', '2.3.1.162', '2.4.1.363', '2.4.1.366', '4.2.3.204', '4.2.3.50', '4.2.3.53', '4.2.3.54', '1.14.14.183']
#
# # try: EC -(BRENDA)-> UniProt
# still_none_found = []
# for EC in tqdm(failed_ECs):
#     upids = BRENDA_EC_to_UP.get(EC)  # key might not exist
#     if upids is None:
#         still_none_found.append(EC)
#     else:
#         ref_seq = fetch_random_seq_from_uniprots(list(upids))
#         EC_to_refseq = EC_to_refseq.append(pd.DataFrame([[EC, ref_seq]]), ignore_index=True)
#
# print(still_none_found)
# # still_none_found = ['1.13.11.45', '1.14.14.124', '1.14.14.125', '1.14.14.168', '1.14.14.170', '2.3.1.162', '2.4.1.363', '2.4.1.366', '4.2.3.204', '4.2.3.50', '4.2.3.53', '4.2.3.54', '1.14.14.183']
#
# FNAME = f'deduplicated_checkpoint2.pkl'
# with open(OUTDIR + '/' + FNAME, 'wb') as fi:
#     pickle.dump(EC_to_refseq, fi, protocol=pickle.HIGHEST_PROTOCOL)
#
# # try: EC -(UniProt)-> UniProt
# still_still_none_found = []
# for EC in tqdm(still_none_found, desc="EC Progress"):
#     ref_seq = get_refseq_for_ec(EC, uniref=90, random=True)
#     if ref_seq:
#         EC_to_refseq = EC_to_refseq.append(pd.DataFrame([[EC, ref_seq]]), ignore_index=True)
#
# EC_to_refseq[0] = EC_to_refseq[0].apply(lambda ec: EC_to_Eid[ec])
# EC_to_refseq.columns = ['Eid', 'ref_seq']
# EC_to_refseq = EC_to_refseq.set_index('Eid')
#
# print('NO SEQUENCE HITS DURING DE-DUPLICATION:')
# print(still_still_none_found)  # EMPTY
#
# for index in dup_seqs.index:
#     if index in EC_to_refseq.index:  # successfully de-duplicated
#         all_seqs.loc[index, 'ref_seq'] = EC_to_refseq.loc[index, 'ref_seq']
#
# print(f'DUPLICATES IN FIRST-PASS DE-DUPLICATED SET {sum(all_seqs.duplicated("ref_seq"))}')
#
# FNAME = f'deduplicated1.pkl'
# with open(OUTDIR + '/' + FNAME, 'wb') as fi:
#     pickle.dump(EC_to_refseq, fi, protocol=pickle.HIGHEST_PROTOCOL)
#
# # ----------------------------------------------------------------------------------------------------------------------
# # DEDUPLICATION 2
# # ----------------------------------------------------------------------------------------------------------------------
# all_files = ['EC_seqs_from_uniref90_via_BRENDA.pkl', 'EC_seqs_from_uniref90_via_KEGGgenes_0.pkl',
#              'EC_seqs_from_uniref90_via_KEGGgenes_1.pkl', 'EC_seqs_from_uniref90_via_KEGGgenes_10.pkl',
#              'EC_seqs_from_uniref90_via_KEGGgenes_2.pkl', 'EC_seqs_from_uniref90_via_KEGGgenes_3.pkl',
#              'EC_seqs_from_uniref90_via_KEGGgenes_4.pkl', 'EC_seqs_from_uniref90_via_KEGGgenes_5.pkl',
#              'EC_seqs_from_uniref90_via_KEGGgenes_6.pkl', 'EC_seqs_from_uniref90_via_KEGGgenes_7.pkl',
#              'EC_seqs_from_uniref90_via_KEGGgenes_8.pkl', 'EC_seqs_from_uniref90_via_KEGGgenes_9.pkl',
#              'EC_seqs_from_uniref90_via_uniprot_directly.pkl', 'by_hand_Uniref90.pkl', 'deduplicated1.pkl']
#
# all_seqs = pd.DataFrame()
# for file in all_files[:-1]:
#     with open(OUTDIR + '/' + file, 'rb') as fi:
#         all_seqs = all_seqs.append(pickle.load(fi))
#
# with open(OUTDIR + '/' + all_files[-1], 'rb') as fi:
#     dedup1 = pickle.load(fi)
#
# for index in dup_seqs.index:  # swap out the duplicate ones for random seqs from one of their upids
#     if index in dedup1.index:
#         if dedup1.loc[index,'ref_seq'] != (None, None):
#             all_seqs.loc[index, 'ref_seq'] = dedup1.loc[index, 'ref_seq']
#
# # make sure that all ECs have a sequence
# none_index = all_seqs[all_seqs['ref_seq'].apply(lambda seq: seq is None)].index # missing sequences
# for i in none_index:
#     seq = get_refseq_for_ec(Eid_to_EC[i])[1]
#     all_seqs.loc[i,'ref_seq'] = seq
#
# # make sure that all ECs are present
# for i in range(len(EC_to_Eid.keys())):
#     if not (i in all_seqs.index):
#         uniref, seq = get_refseq_for_ec(Eid_to_EC[i])
#         df = pd.DataFrame([[i, seq]])
#         df.columns = ['Eid', 'ref_seq']
#         df = df.set_index('Eid')
#         all_seqs = all_seqs.append(df, ignore_index=True)
#
# all_seqs = all_seqs.sort_index()
#
# dup_seqs = all_seqs[all_seqs.duplicated('ref_seq')].index
# for eid in tqdm(dup_seqs):
#     _, seq = get_refseq_for_ec(Eid_to_EC[eid])
#     all_seqs.loc[eid, 'ref_seq'] = seq
#
# dup_seqs = all_seqs[all_seqs.duplicated('ref_seq')].index  # 6 duplicates -> 3 to change
# for eid in tqdm(dup_seqs):
#     upids = BRENDA_EC_to_UP.get(Eid_to_EC[eid])  # key might not exist
#     if upids is None:
#         print(f'failed for {Eid_to_EC[eid]}')
#     else:
#         ref_seq = fetch_random_seq_from_uniprots(list(upids))
#         all_seqs.loc[eid, 'ref_seq'] = ref_seq
#
# # do the rest by hand -> no more duplicates, no more Nones
# all_seqs.loc[1364,'ref_seq'] = 'MELQYISYFQPTSSVVALLLALVSILSSVVVLRKTFLNNYSSSPASSTKTAVLSHQRQQSCALPISGLLHIFMNKNGLIHVTLGNMADKYGPIFSFPTGSHRTLVVSSWEMVKECFTGNNDTAFSNRPIPLAFKTIFYACGGIDSYGLSSVPYGKYWRELRKVCVHNLLSNQQLLKFRHLIISQVDTSFNKLYELCKNSEDNQGNYPTTTTAAGMVRIDDWLAELSFNVIGRIVCGFQSGPKTGAPSRVEQFKEAINEASYFMSTSPVSDNVPMLGWIDQLTGLTRNMKHCGKKLDLVVESIINDHRQKRRFSRTKGGDEKDDEQDDFIDICLSIMEQPQLPGNNNPSQIPIKSIVLDMIGGGTDTTKLTTIWTLSLLLNNPHVLDKAKQEVDAHFRTKRRSTNDAAAAVVDFDDIRNLVYIQAIIKESMRLYPASPVVERLSGEDCVVGGFHVPAGTRLWANVWKMQRDPKVWDDPLVFRPDRFLSDEQKMVDVRGQNYELLPFGAGRRVCPGVSFSLDLMQLVLTRLILEFEMKSPSGKVDMTATPGLMSYKVIPLDILLTHRRIKPCVQSAASERDMESSGVPVITLGSGKVMPVLGMGTFEKVGKGSERERLAILKAIEVGYRYFDTAAAYETEEVLGEAIAEALQLGLVKSRDELFISSMLWCTDAHADRVLLALQNSLRNLKLEYVDLYMLPFPASLKPGKITMDIPEEDICRMDYRSVWAAMEECQNLGFTKSIGVSNFSCKKLQELMATANIPPAVNQVEMSPAFQQKKLREYCNANNILVSAISVLGSNGTPWGSNAVLGSEVLKKIAMAKGKSVAQVSMRWVYEQGASLVVKSFSEERLRENLNIFDWELTKEDHEKIGEIPQCRILSAYFLVSPNGPFKSQEELWDDEA'
# all_seqs.loc[184,'ref_seq'] = 'MENTQRSVIVTGGGSGIGRAVARAFAARGDRVLVVGRTAGPLAETVDGHKEAHTLAVDITDPAAPQAVVREVRERLGGVVDVLVNNAATAVFGHLGELDRTAVEAQVATNLVAPVLLTQALLDPLETASGLVVNIGSAGALGRRAWPGNAVYGAAKAGLDLLTRSWAVELGPRGIRVIGVAPGVIETGAGVRAGMSQEAYDGFLEAMGQRVPLGRVGRPEDVAWWVVRLADPEAAYASGAVLAVDGGLSVT'
# all_seqs.loc[769,'ref_seq'] = 'MSLNMFWFLPTHGDGHYLGTEEGSRPVDHGYLQQIAQAADRLGYTGVLIPTGRSCEDAWLVAASMIPVTQRLKFLVALRPSVTSPTVAARQAATLDRLSNGRALFNLVTGSDPQELAGDGVFLDHSERYEASAEFTQVWRRLLQRETVDFNGKHIHVRGAKLLFPAIQQPYPPLYFGGSSDVAQELAAEQVDLYLTWGEPPELVKEKIEQVRAKAAAHGRKIRFGIRLHVIVRETNDEAWQAAERLISHLDDETIAKAQAAFARTDSVGQQRMAALHNGKRDNLEISPNLWAGVGLVRGGAGTALVGDGPTVAARINEYAALGIDSFVLSGYPHLEEAYRVGELLFPLLDVAIPEIPQPQPLNPQGEAVANDFIPRKVAQS'
#
# with open(OUTDIR + '/' + 'all_seqs.pkl', 'wb') as fi:
#     pickle.dump(all_seqs, fi, protocol=pickle.HIGHEST_PROTOCOL)

# ----------------------------------------------------------------------------------------------------------------------
# ANALYSIS
# ----------------------------------------------------------------------------------------------------------------------

with open(OUTDIR + '/' + 'all_seqs.pkl', 'rb') as fi:
    all_seqs = pickle.load(fi)

import matplotlib.pyplot as plt
# histogram
sequence_lengths = all_seqs['ref_seq'].apply(len)
fig = plt.figure(figsize=(8, 6), dpi=300)
plt.hist(sequence_lengths, bins=150)  # Adjust the number of bins as needed
plt.ylim(0, 600)
# delimiter lines
plt.axvline(x=50, color='red', linestyle='--')
plt.axvline(x=1022, color='red', linestyle='--')
plt.text(-120, 500, 'minimum length:    50 AA', color='red', rotation='vertical', verticalalignment='top')
plt.text(852, 500, 'embedder maximal length:    1022 AA', color='red', rotation='vertical', verticalalignment='top')
# compute stuff
mean_length = sequence_lengths.mean()
percentage_below_1022 = (sequence_lengths < 1022).mean() * 100
# textbox
plt.text(550, 550, "{:.2f}% \nof seqs".format(percentage_below_1022), ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))
# mean-line
plt.xticks(list(plt.xticks()[0]) + [mean_length])
plt.axvline(x=mean_length, color='orange', linestyle='-')
plt.xlim(-400, max(sequence_lengths))
# labels
plt.xlabel('Sequence Length [AA]')
plt.ylabel('Number of Sequences')
plt.title('Histogram of Sequence Lengths', fontsize=14)
# inset
inset_ax = plt.axes([0.45, 0.4, 0.4, 0.4], frame_on=True)
inset_ax.hist(sequence_lengths[sequence_lengths > 1022], bins=50, color='green')
inset_ax.set_xlim(900, max(sequence_lengths))
inset_ax.set_ylim(0, 30)  # Adjust the y-axis limit as needed
inset_ax.axvline(x=1022, color='red', linestyle='--')
inset_ax.text(4500, 27.5, "{:.2f}% of seqs".format(100-percentage_below_1022), ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))
inset_ax.set_xlabel('Sequence Length [AA]')
inset_ax.set_ylabel('Number of Sequences')
# save
plt.savefig(OUTDIR+'/seqs_histo.png', bbox_inches='tight')

# once you have one batch of seqs (assuming it runs without hickup)
# - read everything back in and evaluate:
#
# - are they really all above 50 AA? what is the length distribution?

# once you have some sequences, start playing around with embedding script -> TODAY
