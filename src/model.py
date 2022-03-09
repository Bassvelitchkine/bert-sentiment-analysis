import torch
import numpy as np
import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, inner_dimension=256, nb_context_words=2, nb_last_layers=12, nb_labels=3, seed=0):
        super().__init__()
        self.random_generator_ = np.random.default_rng(seed=seed)
        self.nb_context_words_ = nb_context_words

        self.embeddings_computer_ = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = self.embeddings_computer_.config.hidden_size

        # For each sample, we'll use the the embedding of the center word (i.e the word of interest)
        # As well as nb_context_words sampled from the sentence
        self.base_linear_ = nn.Linear((nb_context_words + 1) * hidden_size, inner_dimension)

        # The last layer depends on the category we're dealing with
        # Look at the forward method for a better gist of what goes on
        for i in range(nb_last_layers):
          setattr(self, f'linear_{i}_', nn.Linear(inner_dimension, nb_labels))

    def forward(self, encoded_sentences, position_indexes, categories):
        '''
        The forward pass of the classifier
        '''
        # Embeddings have the following shape: (sentences_per_batch, nb_tokens_per_sentence, embedding_dimension)
        embeddings = self.embeddings_computer_(encoded_sentences).last_hidden_state
        random_indexes = self.random_generator_.choice(np.arange(embeddings.shape[1]), size=self.nb_context_words_)
        context_embeddings = embeddings[:, random_indexes, :]

        starts = position_indexes[:, 0]
        stops = position_indexes[:, 1]
        output = []

        for sentence_index, sentence_embeddings in enumerate(embeddings):
            start, stop = starts[sentence_index], stops[sentence_index]
            center_word_embedding = torch.mean(sentence_embeddings[start:stop, :], dim=0)
            feature_vector = torch.cat((center_word_embedding, *context_embeddings[sentence_index, :]), 0)

            # We take the layer that corresponds to the category associated to the sample
            specific_layer = getattr(self, f'linear_{categories[sentence_index]}_')

            out = torch.relu(self.base_linear_(feature_vector))
            out = specific_layer(out)
            output.append(out)

        return torch.stack(output)

if __name__ == "__main__":
    from dataset import CustomDataset
    from loader import CustomDataLoader
    from transformers import BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = CustomDataset('./data/traindata.csv', tokenizer)
    loader = CustomDataLoader(dataset=dataset, shuffle=True, batch_size=5).loader
    model = BertClassifier()
    
    sentences, position_indexes, categories, labels = next(iter(loader))
    res = model(sentences, position_indexes, categories)

    print(res)
    print(res.shape)