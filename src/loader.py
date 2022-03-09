import torch
from torch.utils.data import DataLoader

class CustomDataLoader:
    '''
    A function to load a data set with a specific collate function.
    In the data set, there are multiple sentences, each of different lenghts. Consequently,
    they've all been encoded in series of token ids that each have a different length as well.

    Since we're working with pytorch, we're working with tensors. But we can't build tensors from
    sequences of various lengths. For instance, torch.tensor([[1, 2], [1, 2, 3, 4, 5]]) won't work.

    That's why, within a batch of encoded sentences, they all need to have the same length.

    Here's the basic idea. Let's assume that we have batches of size two and a first batch with:
    - A sentence encoded as [1, 2]
    - A second sentence encoded as [1, 2, 3, 4, 5]

    We want to pad the first sentence, like so: [1, 2, 0, 0, 0], assuming that the padding token
    id is 0. So that we can build a tensor from [[1,2, 0, 0, 0], [1, 2, 3, 4, 5]].
    '''

    def __init__(self, **kwargs) -> None:
        dataset = kwargs['dataset']
        self.pad_token_id_ = dataset.helper_token_ids['pad']
        self.loader = DataLoader(**kwargs, collate_fn=self.__collate_function)

    def __collate_function(self, batch):
        '''
        The private method that builds data batches.

        In each batch there are samples, and a sample is:
        (encoded_sentence, position_indexes, category, label)
        '''
        longest_sentence_length = max([len(sample[0]) for sample in batch])

        encoded_sentences = []
        position_indexes = []
        categories = []
        labels = []

        for sample in batch:
            encoded_sentence, pos_indexes, category, label = sample
            sentence_length = len(encoded_sentence)
            new_sentence = encoded_sentence + (longest_sentence_length - sentence_length) * [self.pad_token_id_]

            encoded_sentences.append(new_sentence)
            position_indexes.append(pos_indexes)
            categories.append(category)
            labels.append(label)

        return torch.tensor(encoded_sentences), torch.tensor(position_indexes), torch.tensor(categories), torch.tensor(labels)

if __name__ == "__main__":
    from dataset import CustomDataset
    from transformers import BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = CustomDataset('./data/traindata.csv', tokenizer)
    loader = CustomDataLoader(dataset=dataset, shuffle=True, batch_size=5).loader
    
    sentences, position_indexes, categories, labels = next(iter(loader))

    print(sentences)
    print(position_indexes)
    print(categories)
    print(labels)