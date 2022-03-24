# Authors

Bastien Vélitchkine, Victoire Tusseau, Antonin Gagneré and Simon Rouard.

# Problem description

Here's what the two first lines of the training data set look like:

| label    | hashtag          | center_word | center_word_position | text                                  |
| -------- | ---------------- | ----------- | -------------------- | ------------------------------------- |
| positive | AMBIENCE#GENERAL | seating     | 18:25                | short and sweet – seating is great... |
| positive | AMBIENCE#GENERAL | trattoria   | 25:34                | This quaint and romantic trattoria... |

It's a data set of restaurant reviews. The first column classifies the comment either as _negative_, _neutral_ or _positive_. The second column is a hashtag associated to the comment (it somehow gives a gist of what the comment is specifically about), the third column is for center words (the one that was classified either as neg, neutr or pos), then the indexes of the beginning and ending of the center word, and finally, the full text of the comment in the fifth column.

We're obviously facing a problem of **sentiment analysis**. We want to be able to **classify specific words** in a comment either as _negative_, _neutral_ or _positive_.

Our metric for this problem is the **accuracy** on a test data set, very similar to our training data set. We want to reach the highest accuracy.

# Model

## General idea

We roamed a bit on the internet to try and find the state of the art when it comes to NLP and sentiment analysis. It turns out that some of the best models so far are models that **properly map words and sentences to their vector representation**.

This very mapping is the first and biggest challenge of natural language processing, since mathematical models can't understand plain words.

Once we have proper feature vectors to represent our input sentences, the rest is pretty straight-forward: Classic classifiers can be used (whether deep or traditional, like SVMs).

After a bit of research, especially on [huggingface.co](https://huggingface.co/), it turned out that the _BERT model_ - developed by Google - was an excellent solution to base our classifier on.

Hence, the outline of our model:

- Map our input data (words) to their vectorial embeddings thanks to _BERT_
- Use combinations of these embeddings as inputs for our deep learning classifier.

> Note that **we did not have to train BERT from scratch**, just use its precomputed embeddings, built after hours of training on extensive data by Google teams.

Let's dive in 👇

## Tokenization

The first step is to tokenize words in the training sentences i.e to map them to a _first unique numeric representation_.

> This tokenization step was performed with huggingface's [BERT tokenizer](https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer).

## Vectorization

Tokens were then used as inputs for huggingface's [BERT model](https://huggingface.co/docs/transformers/v4.17.0/en/model_doc/bert#transformers.BertModel) and transformed in proper embeddings i.e vectors with a few hundred dimensions.

## Classification

The classifier is built as follows:

1. A base `torch.nn.Linear` layer, common to any embedding passed as input
2. Twelve `torch.nn.Linear` layers, one for each of the possible _hashtags_ presented in the data set above.

Example:

If we had a comment in the data set that was associated to the hashtag AMBIENCE#GENERAL, this sentence would first go through the base layer, and then through one of the 12 layers of the second step, and **one only**.

There are 12 such hashtags, hence 12 such layers. This adds more granularity to the model and seems like an efficient way to **leverage the extra hashtag data**.

## Optimization

The criterion to optimize was `torch.nn.CrossEntropyLoss`, a loss that is commonly used in multi-label classification.

As an optimizer, we used `torch.optim.Adam`, one of the first that we tried, and seemed pretty good from the start.

# Code organization

We created four files in addition to the `classifier.py` and `tester.py` files: `dataset.py`, `loader.py` and `model.py`.

Let us walk you through each one of them.

> Note that we tried not to set astray from pytorch's usual objects and training steps. Most projects based on pytorch have similar architectures.

# Dataset

`dataset.py` contains a single `CustomDataset` class built upon `torch.utils.data.Dataset`. In this class here's what we did:

1. We retrieved data from the training data set (a csv file) and put it in a pandas dataframe.

2. We replaced the hashtags (SERVICE#GENERAL, FOOD#QUALITY, etc.) with unique identifiers. Therefore, we have 12 distinct identifiers for the 12 hashtags (as stated previously). We call them _Categories_.

3. We also replaced the opinions (_positive_, _neutral_ and _negative_) with unique numerical identifiers. We call them _Labels_

4. We **tokenized our sentences with the BERT tokenizer**, and we retrieved at the same time the new position of the target token (the token of the target word). They're the _encoded sentences_.
   > Indeed, one might ask why the position of the center word would change if each word of the sentence is transformed in a single token. Well that's the trick: Some words can be mapped to multiple tokens at once and change the length of the sentence in the tokenized space, hence the new positions.

All in all, our data set encompasses 4 sequences:

- Encoded sentences
- Position indexes
- Categories
- Labels.

> Usually, data sets have inputs and labels. In our case, the three first entries constitue the inputs and the last one, "labels", the labels like in any classification problem.

## Data loader

The `loader.py` file contains a single class as well, a `CustomDataLoader` built upon `torch.utils.data.DataLoader` to batch the data set data.

The specificity of this class lies in its _collate function_, the method used to batch the inputs and transform them into tensors. In this _collate function_, we padded each sentence so that they all have the same length (since we cannot build tensors from lists of different lengths)

## Model

`model.py` contains the 2-layer classification model. As stated above, it's made of:

1. A base linear layer
2. One of twelve specific linear layers, one for each _category_.

Instead of feeding our raw embeddings to the classifier, we built more complex inputs from these very embeddings. As we said before, there's a single center word that we want to labelize either as positive, neutral or negative. But one single word is often not enough to make a fine-grained prediction: Context words (words surrounding the center word) have a role to play as well.

In mathematical terms, it translates as follows:

1.  We take the embedding of the center word
2.  We randomly choose to context words and take their embeddings as well.
    > Note that a way to improve the algorithm would therefore be to increase the number of context words (which would undoubtedly increase computational time).
3.  We concatenate the three selected embeddings into a single and large feature vector.

**A subtlety**:

As we said before, a single word can sometimes be **translated into multiple tokens**, first by the _BERT Tokenizer_, and then into multiple embeddings by the _BERT Model_. That's why we averaged the **potentially multiple embeddings of the single center word**.

The concatenated vector is then fed to the two layers of the
neural networks (by batches though, like always in pytorch). Therefore, if a single embedding has dimension $128$, our feature vector will have dimension $3 * 128 = 384$.

## Classifier

The classifier, encompassed in `classifier.py` is not the classifier per say, but rather **a wrapper** that contains the model, and a few methods to train and test the model on the data.

Let's focus on the training.

As mentioned above, we used Adam's optimizer on the cross entropy loss. What's new is the `get_linear_schedule_with_warmup` function from huggingface, used as scheduler for the learning rate. Indeed, to put things simply, we often want the learning rate to start big and wittle down as we run through an increasing number of epochs - as we move from tuning to fine-tuning. This function builds a "warmed-up scheduler", one whose learning rate was allegedly smartly initialized and will intelligently decrease. This is another way to improve our performances, in a minimum number of training steps.

# Results

We reached an **accuracy that varies between 75% and 85%**, depending on the chosen random seed. Since we did not have that much data, and not that much time to increase our performances, this seems like pretty decent results.

We could also increase the number of layers in our neural network but that would expose us to overfitting. It would then be necessary to drop out neurons in the upper layers. The deeper a
neural network, the more expressive it is. But given the size of our dataset, we preferred to stick to two mere layers, as explaned before.
