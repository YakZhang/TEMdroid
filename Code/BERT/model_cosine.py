#coding = utf-8

import torch
import torch.nn as nn
from transformers import (
    BertConfig,
    BertTokenizer,
    BertModel,
    BertPreTrainedModel
)
from transformers.modeling_outputs import MultipleChoiceModelOutput, TokenClassifierOutput
from config import ModelArguments
import os
import json

def flat(t: torch.Tensor):
    return t.view(-1, t.shape[-1]) if t is not None else None


def avg_pooling(embs, input_ids, start_tok: int, end_tok: int):
    start = torch.nonzero(input_ids == start_tok, as_tuple=True)[1]
    end = torch.nonzero(input_ids == end_tok, as_tuple=True)[1]
    result = []
    for i, (s, e, item) in enumerate(zip(start + 1, end, embs)):
        result.append(torch.mean(item[s: e], dim=0))
    return torch.vstack(result)




class MultipleChoices_network(BertPreTrainedModel):
    # the init model is from huggface network api call
    _keys_to_ignore_on_load_missing = [r'position_ids']
    
    def __init__(self, model_args, config):
    # def __init__(self, plm, config):
        super().__init__(config)
        self.bert = BertModel.from_pretrained(
            model_args.model_name_or_path,
            config = config,
            cache_dir = model_args.cache_dir,
            revision  = model_args.model_revision
        )
        # self.bert = plm
        hid = 700
        input = 768


        self.layer1 = nn.Linear(input,hid)
        self.layer2 = nn.Linear(hid,hid)
        
    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            other_features = None,
            labels=None,
            nl_feature_tokens = None,

    ):
        

        


        outputs = self.bert(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states = True, # optional for hidden_states
        )
        
        pooler_outs = outputs.pooler_output
        

        pooler_shape = pooler_outs.shape

        pooler_outs = pooler_outs.view(-1,2,pooler_shape[1])
        pooler_first = pooler_outs[:,0,:]
        pooler_second = pooler_outs[:,1,:]

        features_first = pooler_first
        features_second = pooler_second

        

        

        x1 = torch.relu(self.layer1(features_first.float()))
        x2 = torch.relu(self.layer1(features_second.float()))


        x1 = torch.tanh(self.layer2(x1))
        x2 = torch.tanh(self.layer2(x2))
        
        # x1 = self.layer2(x1)
        # x2 = self.layer2(x2)

        logits = torch.nn.functional.cosine_similarity(x1,x2,-1)



        
        


        loss = torch.mean((logits - labels) ** 2) if labels is not None else None
        # return MultipleChoiceModelOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
        return TokenClassifierOutput(
            loss = loss,
            logits = logits,
            # attentions = features,
        )
        # return logits, loss, features
    





class MultipleChoices(nn.Module):

    _keys_to_ignore_on_load_missing = [r'position_ids']
    
    # for download plm
    def __init__(self, plm):
        super().__init__()
        self.bert = plm
    

        # for text_feature+ other_feature
        hid = 700
        input = 768
        # input = 780
        # input = 768+(9*10)

        self.layer1 = nn.Linear(input,hid)
        self.layer2 = nn.Linear(hid,hid)

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            other_features = None,
            labels=None,
            nl_feature_tokens = None,

    ):



        outputs = self.bert(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states = True, # optional for hidden_states
        )
        
        pooler_outs = outputs.pooler_output
        


        pooler_shape = pooler_outs.shape
        if pooler_shape[0]==43:
            print("here")
        pooler_outs = pooler_outs.view(-1,2,pooler_shape[1])
        pooler_first = pooler_outs[:,0,:]
        pooler_second = pooler_outs[:,1,:]

        features_first = pooler_first
        features_second = pooler_second


        

        x1 = torch.relu(self.layer1(features_first.float()))
        x2 = torch.relu(self.layer1(features_second.float()))

        x1 = torch.tanh(self.layer2(x1))
        x2 = torch.tanh(self.layer2(x2))


        logits = torch.nn.functional.cosine_similarity(x1,x2,-1)



        


        loss = torch.mean((logits - labels) ** 2) if labels is not None else None
        # return MultipleChoiceModelOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
        return TokenClassifierOutput(
            loss = loss,
            logits = logits,
            # attentions = features,
        )
        # return logits, loss, features

        
    


def load_plm(cache_dir,logger):
    logger.info('start loading model and tokenizer ...')
    print('start loading model and tokenizer...')
    model: BertModel = torch.load(os.path.join(cache_dir, 'model.pth'))
    tokenizer: BertTokenizer = torch.load(os.path.join(cache_dir, 'tokenizer.pth'))
    logger.info('done loading model and tokenizer ...')
    print('done loading model and tokenizer ... ')
    return model, tokenizer

def load_config(cache_dir, logger):
    logger.info("start loading config")
    print("start loading config")
    with open(os.path.join(cache_dir,'config.json')) as f:
        config_json= json.load(f)
    logger.info('don loading config')
    print('done loading config')
    return config_json


def init_model(logger,model_args:ModelArguments):
    if model_args.load_plm_path == None:
        if model_args.load_cfg_path == None:
            config = BertConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision
            )
            model = MultipleChoices_network(model_args,config)
            tokenizer = BertTokenizer.from_pretrained(
                model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                use_fast=model_args.use_fast_tokenizer,
                revision=model_args.model_revision
            )
            return model, tokenizer
        else:
            config_json = load_config(model_args.load_cfg_path, logger)
            config = BertConfig(
                attention_probs_dropout_prob=config_json["attention_probs_dropout_prob"],
                hidden_act=config_json["hidden_act"],
                hidden_dropout_prob=config_json["hidden_dropout_prob"],
                hidden_size=config_json["hidden_size"],
                initializer_range=config_json["initializer_range"],
                intermediate_size=config_json["intermediate_size"],
                max_position_embeddings=config_json["max_position_embeddings"],
                num_attention_heads=config_json["num_attention_heads"],
                num_hidden_layers=config_json["num_hidden_layers"],
                type_vocab_size=config_json["type_vocab_size"],
                vocab_size=config_json["vocab_size"])
            model = MultipleChoices_network(model_args,config)

            tokenizer = BertTokenizer.from_pretrained(
                model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                use_fast=model_args.use_fast_tokenizer,
                revision=model_args.model_revision
            )
            return model, tokenizer
    else:
        plm, tokenizer = load_plm(model_args.load_plm_path,logger)
        # config = BertConfig.from_pretrained(
        #     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        #     cache_dir=model_args.cache_dir,
        #     revision=model_args.model_revision
        # )
        config_json = load_config(model_args.load_cfg_path, logger)
        config = BertConfig(
                attention_probs_dropout_prob=config_json["attention_probs_dropout_prob"],
                hidden_act=config_json["hidden_act"],
                hidden_dropout_prob=config_json["hidden_dropout_prob"],
                hidden_size=config_json["hidden_size"],
                initializer_range=config_json["initializer_range"],
                intermediate_size=config_json["intermediate_size"],
                max_position_embeddings=config_json["max_position_embeddings"],
                num_attention_heads=config_json["num_attention_heads"],
                num_hidden_layers=config_json["num_hidden_layers"],
                type_vocab_size=config_json["type_vocab_size"],
                vocab_size=config_json["vocab_size"])
        model = MultipleChoices_network(model_args,config)
        # model = MultipleChoices_network(plm,config)

        return model, tokenizer
