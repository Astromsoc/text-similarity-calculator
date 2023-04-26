"""
    Models (from huggingface) in use to obtain embeddings or compute similarity directly.


    ---

    Last updated:
        Apr 25, 2023 

"""


import torch
import torch.nn as nn


from typing import List
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer




class TSCalculatorFromTransformersInferenceOnly:

    def __init__(self,
                 tokenizer: str='sentence-transformers/all-mpnet-base-v2',
                 model: str='sentence-transformers/all-mpnet-base-v2',
                 eps: float=1e-9):
        # archiving
        self.tokenizer_opt  = tokenizer
        self.model_opt      = model
        self.eps            = eps
        # build tokenizer & model
        self._init_models()
        
    
    def _init_models(self):
        self.tokenizer  = AutoTokenizer.from_pretrained(self.tokenizer_opt)
        self.model      = AutoModel.from_pretrained(self.model_opt)
        # set model to eval model
        self.model.eval()

    
    @staticmethod
    def get_mean_pool(model_outputs, attention_mask, eps):
        # (batch_size, max_len, emb_dim)
        token_embeddings = model_outputs[0]
        # (batch_size, max_len, emb_dim)
        expanded_masks = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * expanded_masks).sum(dim=1) / torch.clamp(expanded_masks.sum(dim=1), min=eps)


    @staticmethod
    def get_cos_similarity(sentemb1: torch.tensor, sentemb2: torch.tensor):
        return nn.functional.cosine_similarity(sentemb1[None, :], sentemb2[None, :])


    def get_embeddings(self, sentences: List[str]):
        """
            Obtain the sentence embedding of inputs.

            Args:
                sentence (List[str]): sentence to be converted
            
            Returns:
                (torch.tensor) Sentence embeddings (B, emb_dim)
        """
        # text --> token ids + attention masks
        tokens = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        """
            input_ids: (batch_size, max_seq_len)
            token_type_ids: (batch_size, max_seq_len)
            attention_masks: (batch_size, max_seq_len)
        """
        # token ids --> token embeddings
        with torch.inference_mode():
            token_embeddings = self.model(**tokens)
        """
            model_outputs (index 0): (batch_size, max_len, emb_dim)
        """
        # token embeddings --> sentence embeddings
        return self.get_mean_pool(model_outputs=token_embeddings,
                                  attention_mask=tokens['attention_mask'],
                                  eps=self.eps)




class TSCalculatorFromSentenceTransformersInferenceOnly:

    def __init__(self,
                 model: str='sentence-transformers/all-mpnet-base-v2',
                 eps: float=1e-9):
        # archiving
        self.model_opt      = model
        self.eps            = eps
        # build tokenizer & model
        self._init_model()
    

    def _init_model(self):
        self.model = SentenceTransformer(self.model_opt)
    

    @staticmethod
    def get_cos_similarity(sentemb1: torch.tensor, sentemb2: torch.tensor):
        return nn.functional.cosine_similarity(sentemb1[None, :], sentemb2[None, :])

    
    def get_embeddings(self, sentences: List[str]):
        return self.model.encode(sentences, convert_to_tensor=True)




"""
    Model type switches
"""
ModelTypeDict = {
    'transformers': TSCalculatorFromTransformersInferenceOnly,
    'sentence-transformers': TSCalculatorFromSentenceTransformersInferenceOnly
}