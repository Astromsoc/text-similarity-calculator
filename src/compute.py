"""
    Driver script to run the similarity calculation.

    Currently target granularity:
        [1] sentence

    ---

    Last updated:
        Apr 25, 2023 

"""

import os
import pickle
import argparse

from tqdm import tqdm
from ruamel.yaml import YAML
yaml = YAML(typ='safe')


from src.utils import *
from src.models import *



def main(args):
    # load configs
    configs = ParamsObject(yaml.load(open(args.configs, 'r')))

    # build inferer
    inferer = ModelTypeDict.get(configs.model.type, 'transformers')(**configs.model.configs.__dict__)

    # load data
    sentence_groups, key_orders = load_sentences(configs.src_filepath)
    num_sentences = len(key_orders)
    compound_results = list() if isinstance(sentence_groups, list) else dict()
    
    # compute similarity group by group
    for sentences in tqdm(sentence_groups.items() if isinstance(sentence_groups, dict) else sentence_groups, 
                          desc=f'Iterating through [{len(sentence_groups)}] cases...'):
        # unpack case id
        if isinstance(sentences, tuple): cid, sentences = sentences
        # obtain total number of sentences in the current group
        # obtain sentence embeddings: (batch_size, emb_dim)
        sentence_embs = inferer.get_embeddings(sentences=sentences)
        cos_similarity_scores = torch.ones((num_sentences, num_sentences))
        for i in range(num_sentences):
            for j in range(i + 1, num_sentences):
                cos_similarity_scores[i, j] = inferer.get_cos_similarity(
                    sentence_embs[i], sentence_embs[j]
                )
                cos_similarity_scores[j, i] = cos_similarity_scores[i, j]
        # store the results
        result = {'sentence_embeddings': {k: sentence_embs[i] for i, k in enumerate(key_orders)},
                  'cos_similarity': cos_similarity_scores,
                  'key_orders': key_orders}
        if isinstance(compound_results, dict):
            compound_results[cid] = result
        else:
            compound_results.append(result)

        # save the output file in every iteration (in case of crashes)
        pickle.dump(compound_results, open(configs.out_filepath.format(configs.model.type), 'wb'))




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Text similarity calculation.")
    parser.add_argument('--configs', '-c', type=str, help="(str) Filepath to configurations.")
    args = parser.parse_args()

    assert os.path.exists(args.configs), f"[** FILE NOT EXISTED **] Check if ({args.configs}) does exist."

    main(args)