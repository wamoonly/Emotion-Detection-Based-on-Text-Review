from torch.utils.data import Dataset
import torch

class dataset_ATM(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        tokens, tags = self.df[idx][:2]  # Directly index the list instead of using iloc

        tokens = tokens.replace("'", "").strip("][").split(', ')
        tags = tags.strip('][').split(', ')

        bert_tokens = []
        bert_tags = []
        for i in range(len(tokens)):
            t = self.tokenizer.tokenize(tokens[i])
            bert_tokens += t
            bert_tags += [int(tags[i])] * len(t)

        bert_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)

        ids_tensor = torch.tensor(bert_ids)
        tags_tensor = torch.tensor(bert_tags)

        return bert_tokens, ids_tensor, tags_tensor

    def __len__(self):
        return len(self.df)


# from torch.utils.data import Dataset
# import torch

# class dataset_ATM(Dataset):
#     def __init__(self, df, tokenizer):
#         self.df = df
#         self.tokenizer = tokenizer

#         # Add special tokens for words that should not be split
#         special_tokens = ['nasi', 'lemak']
#         self.special_tokens_ids = self.tokenizer.convert_tokens_to_ids(special_tokens)

#     def __getitem__(self, idx):
#         tokens, tags = self.df[idx][:2]  # Directly index the list instead of using iloc

#         tokens = tokens.replace("'", "").strip("][").split(', ')
#         tags = tags.strip('][').split(', ')

#         bert_tokens = []
#         bert_tags = []
#         for i in range(len(tokens)):
#             if tokens[i] in self.tokenizer.get_vocab() and self.tokenizer.convert_tokens_to_ids(tokens[i]) in self.special_tokens_ids:
#                 # Add whole token for special words
#                 bert_tokens.append(tokens[i])
#                 bert_tags.append(int(tags[i]))
#             else:
#                 t = self.tokenizer.tokenize(tokens[i])
#                 bert_tokens += t
#                 bert_tags += [int(tags[i])] * len(t)

#         bert_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)

#         ids_tensor = torch.tensor(bert_ids)
#         tags_tensor = torch.tensor(bert_tags)

#         return bert_tokens, ids_tensor, tags_tensor

#     def __len__(self):
#         return len(self.df)


