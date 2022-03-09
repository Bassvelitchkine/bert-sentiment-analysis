import pandas as pd
from torch.utils.data import Dataset

COLUMNS = ['opinion', 'categories', 'word', 'position', 'sentence']

class CustomDataset(Dataset):
    '''
    A class to load the data from a file
    '''
    def __init__(self, filepath, tokenizer, category_to_id_mapper=None, label_to_id_mapper=None) -> None:
        super().__init__()
        self.tokenizer_ = tokenizer

        # Load the data
        self.data_ = pd.read_csv(filepath, sep='\t', names=COLUMNS)

        # Encode categories if need be
        if category_to_id_mapper is None:
            unique_categories = self.data_['categories'].unique()
            self.category_to_id_mapper = {category: id for id, category in enumerate(unique_categories)}
        else:
            self.category_to_id_mapper = category_to_id_mapper
        
        # Encode labels if need be
        if label_to_id_mapper is None:
            unique_labels = self.data_['opinion'].unique()
            self.label_to_id_mapper = {label: id for id, label in enumerate(unique_labels)}
        else:
            self.label_to_id_mapper = label_to_id_mapper

        # Instantiate reverse mappers, from id to textual values
        self.id_to_label_mapper = {v: k for k, v in self.label_to_id_mapper.items()}
        self.id_to_category_mapper = {v: k for k, v in self.category_to_id_mapper.items()}

        # We instantiate the preprocessed data
        self.encoded_sentences_, self.position_indexes_, self.categories_, self.labels_ = self.__data_processing()

    def __data_processing(self):
        '''
        A private method that processes the input data so that it can later be dealt with
        '''
        # Have start and end indexes in specific columns
        self.data_[["start", "stop"]] = self.data_['position'].str.split(":", expand=True).astype(dtype='int32')

        # We change textual values to their ids
        categories = self.data_['categories'].apply(lambda category: self.category_to_id_mapper[category]).array
        labels = self.data_['opinion'].apply(lambda label: self.label_to_id_mapper[label]).array

        # Hugging face's tokenizers have special token for:
        # - the beggining of a sentence
        # - the end of a sentence
        # - padding
        # Let's recover them:
        sentence_begin_token_id, sentence_start_token_id, pad_token_id = self.tokenizer_("", return_token_type_ids=False, return_attention_mask=False, padding="max_length", max_length=3)['input_ids']
        self.helper_token_ids = {
            "sentence_begin": sentence_begin_token_id,
            "sentence_end": sentence_start_token_id,
            "pad": pad_token_id
        }

        # We encode sentences
        encoded_sentences = []
        position_indexes = []
        for _, (sentence, start, stop), in self.data_[['sentence', 'start', 'stop']].iterrows():
            sentence_token_ids, start_index, stop_index = self.__custom_tokenizer(sentence, start, stop)
            encoded_sentences.append(sentence_token_ids)
            position_indexes.append([start_index, stop_index])

        return encoded_sentences, position_indexes, categories, labels


    def __custom_tokenizer(self, sentence, start_pos, stop_pos):
        '''
        A private method that leverages the input tokenizer to tokenize every sentence.
        
        Why use this method an not the tokenizer directly on the full sentence?

        Well, first of all, most sentences have more token ids once they're encoded than they had
        words in the beginning. Try ```tokenizer.encode("Hi there!")```

        We have start and stop position for a single word in each sentence. But if the length of the
        encoded sentence changes (and it's the case), those positions don't mean anything anymore.

        So in the process of encoding the sentence, we have to recompute these start and stop positions.

        Note that, when encoding a portion of a sentence, we always remove starting and trailing token ids,
        they're the ids that indicate the beggining and end of a sentence. We manualy append and prepend them.
        '''
        sentence_begin, sentence_end = self.helper_token_ids['sentence_begin'], self.helper_token_ids['sentence_end']
        sentence_token_ids = [sentence_begin]

        # Tokenize first part of the sentence, before the word of interest
        if start_pos > 0: 
            sentence_token_ids += self.tokenizer_.encode(sentence[:start_pos], add_special_tokens=True, return_token_type_ids =False, return_attention_mask =False)[1:-1]
        
        # Tokenize the word of interest and store the positions in the process
        start_index = len(sentence_token_ids)
        sentence_token_ids += self.tokenizer_.encode(sentence[start_pos:stop_pos], add_special_tokens=True, return_token_type_ids =False, return_attention_mask =False)[1:-1]
        stop_index = len(sentence_token_ids)

        # Tokenize the last part of the sentence, after the word of interest
        if stop_pos < len(sentence):
            sentence_token_ids += self.tokenizer_.encode(sentence[stop_pos:], add_special_tokens=True, return_token_type_ids =False, return_attention_mask =False)[1:-1]

        sentence_token_ids.append(sentence_end)

        return sentence_token_ids, start_index, stop_index

    # Now we override torch.utils.data.DataSet built-in methods
    def __getitem__(self, index) :
        return self.encoded_sentences_[index], self.position_indexes_[index], self.categories_[index], self.labels_[index]
        
    def __len__(self):
        return self.labels_.shape[0]

if __name__ == "__main__":
    from transformers import BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = CustomDataset('./data/traindata.csv', tokenizer)

    print(dataset.encoded_sentences_[:5])
    print(dataset.position_indexes_[:5])
    print(dataset.categories_[:5])
    print(dataset.labels_[:5])