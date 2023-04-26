"""
    
    Utility functions.

    ---

    Last updated:
        Apr 25, 2023 

"""


import json

from typing import Dict




class ParamsObject(object):

    def __init__(self, yaml_cfgs: Dict) -> None:
        # current level
        self.__dict__.update(yaml_cfgs)
        # lower level(s)
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                self.__dict__[k] = ParamsObject(v)



def load_sentences(filepath: str):
    """
        Assuming loading sentences from json files.

        Args:
            filepath (str): path to the source file 
        
        Returns:
            res (List[List[str]]): list of sentences in group (each group will be compared)
            ks (List[str]): order of key names that correspond to the list of flattened sentences
    """
    # load all cases
    cases = json.load(open(filepath, 'r'))
    # fix the order of sentence keys
    ks = [k for k in cases[0].keys() if k != 'case_id']
    res = dict() if 'case_id' in cases[0] else list()
    for case in cases:
        arranged_sentences = [case[k] for k in ks]
        if 'case_id' in case: 
            res[case['case_id']] = arranged_sentences
        else: 
            res.append(arranged_sentences)
    return res, ks
