"""
   
    Plotting the similarity graph for sentences.

    ---

    Last updated:
        Apr 25, 2023 

"""

import os
import re
import torch
import pickle
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

from tqdm import tqdm


# MODIFIABLE PARAMS
NUM_COLS    = 5
UNIT_SIDE   = 15
WIDTH       = NUM_COLS * UNIT_SIDE




def main(args):

    # load embeddings & similarity scores
    loaded = pickle.load(open(args.src, 'rb'))
    
    # create output folder
    folder = re.match(r"(.*)(\..*)", args.src).group(1)
    os.makedirs(folder, exist_ok=True)
    # copy the source file for archiving
    os.system(f"cp {args.src} {folder}/src.pkl")
    # output image filename format
    filename = 'cos-similarity-of-sentence-embeddings-{}.png'

    # load all cases
    num_total, isDict = len(loaded), isinstance(loaded, dict)
    # shorten the keynames
    key_orders = (list(loaded.values())[0] if isDict else loaded)['key_orders']
    shorten_key = (lambda x: 'tcot-' + x.split('_')[3] if 'tcof' in x and ('except' in x or 'only' in x) 
                             else 'tcot' if 'tcof' in x 
                             else 'baseline' if x.startswith('gen') 
                             else x.replace('_', '-'))
    key_orders = [shorten_key(k) for k in key_orders]
    # start counting pic nums
    accumulated_counter = 0

    # plot the average case
    plt.figure(figsize=(16, 16))
    plt.title(f"Average Cosine Similarity Matrix for [{folder.rsplit('/', 1)[1]}]")
    sns.heatmap(torch.mean(torch.stack([u['cos_similarity'] 
                                        for u in (loaded.values() if isDict
                                                  else loaded)], dim=0), dim=0),
                vmin=0.0, vmax=1.0, cmap='gray_r', annot=True,
                xticklabels=key_orders, yticklabels=key_orders)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'avg-cos-similarity-of-sentence-embeddings.png'), dpi=128)

    # plot all cases
    for i, unit in tqdm(enumerate(loaded.items() if isDict else loaded), total=num_total,
                                 desc=f"Iterating through [{num_total}] cases..."):
        if isDict: 
            cid, cossim_matrix = unit[0], unit[1]['cos_similarity']
        else:
            cossim_matrix = unit['cos_similarity']
        if i == 0 or i % (NUM_COLS ** 2) == 0:
            # save the previous image
            if accumulated_counter != 0:
                plt.tight_layout()
                plt.savefig(os.path.join(folder, filename.format(accumulated_counter)), dpi=128)
            
            # compute number of rows for new image
            num_rows = NUM_COLS if i + NUM_COLS ** 2 <= num_total else (num_total - i) // NUM_COLS + 1
            num_cols = NUM_COLS if i + NUM_COLS <= num_total else (num_total - i)
            # create new image
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(UNIT_SIDE, UNIT_SIDE), sharex=True, sharey=True)
            start_idx = accumulated_counter * NUM_COLS ** 2 + 1
            fig.suptitle(f"Cosine Similarity Matrix in Cases[{start_idx}-{min(start_idx + NUM_COLS ** 2 - 1, num_total)}]")
            accumulated_counter += 1

        # compute axes row & col counts
        ii = i - accumulated_counter * (NUM_COLS ** 2)
        r, c = ii // NUM_COLS, ii % NUM_COLS

        g = sns.heatmap(cossim_matrix, vmin=0.0, vmax=1.0, cmap='gray_r',
                        xticklabels=list(), yticklabels=list(),
                        ax=axes[r, c] if num_rows > 1 else axes[c])
        # set labels
        g.set_title(f'CaseID=[{cid}]({i + 1}/{num_total})' if isDict else f'Case #{i + 1}')

    # save the final image
    plt.tight_layout()
    plt.savefig(os.path.join(folder, filename.format(accumulated_counter)), dpi=128)

    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Plot the cosine similarity matrix for different cases.")
    parser.add_argument('--src', '-s', type=str, help='(str) Filepath to the output generated by "src/compute.py".')
    args = parser.parse_args()
    assert args.src, f"[** FILE NOT EXISTED **] File [{args.src}] does not exist."

    main(args)