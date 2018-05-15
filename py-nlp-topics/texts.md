# Natural language processing for Topic Modeling


Using python + natural language processing for topic modeling: a unsupervised technique for document classification

https://github.com/erickfis


## Topic modeling

Imagine you have a huge collection of documents, each one talking about a specific matter|subject.
A document could be a movie description, or part of a book, a message, a tweet, etc...


If you take the time to read each document, you will learn that they are talking about science, or politics,
or medicine, or sports, etc, but very often they don't have a label specifying the subject.


Now it's your job to group them by subject. Will you read each one of them and label one by one?
What if your collection contains 1 billion documents?


This is where *Topic modeling* comes in hand: it is a very useful technique for document classification through unsupervised learning. It will learn from the collection of documents as a whole and then suggest groups (clusters) of documents by similarities, such as frequency or probabilities for words on each document.


After the documents are split into the suggested groups, we can then look at each group (through samples of them) and choose a proper label for it.


This clustering of documents by topic|subject can be achieved by different techniques:

- tf-idf + clustering
- tf-idf + PCA
- Latent semantic analysis - LSA
- Latent Dirichlet Allocation - LDA

On this notebook we will discuss the LDA method.

## LDA - Latent Dirichlet Allocation

LDA is a especial case of the Latent semantic analysis, where the priors distribution of topics are assumed to be
of the beta multivariate type, aka. Dirichlet distribution.


The main advantage of LDA over LSI, PCA or regular clustering is that LDA is capable of detecting intermediary topics between the ones that would be detected by the former, as they will work on principal components and detect only orthogonal
topics. Thus, LDA reduces overfitting and increases accuracy.


On the other hand, LDA demands more computation time.


https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation


## Use case: chat bots

Imagine we are a big on line retailer and we want to provide a new communication channel for our costumers.

We had a real time on line chat before and we were using a human attendant to answer our costumers.

We have stored all of the costumers messages on our database and now we want to create an algorithm that is capable of answering costumers just like a human attendant would.

But how to answer our costumers in a natural way, simulating the behavior of a human attendant?

The achieve that goal, we are going to use natural language processing and topic modeling. We will combine different techniques for analyzing the costumers queries and questions to elaborate an appropriate natural answer.

The techniques are:

- topic modeling - to orient the composition of a proper answer according to the detected subject of conversation.
IE: if the subject is a complaint, the algorithm should compose an answer taking this information into account.
If the message is a salutation, then salute back.
- keyword detection, as order number, or some specific product.


## Retrieve stored messages from our database




In order to "train" our algorithm, we must retrieve stored data from our databases.
Note that this is not the typical training process because we don't have any targets or labels ready at this point. This is, in fact, a clustering problem.

If the database in question is a NoSQL, like MongoDB, we could run queries like:

    db.mensagens.find(
        {
        $and: [
                {"user": {$in: ["user_1", "user_2", "attd_1", "sell_1"]}},
                {"message": {$ne: null}}
              ]
        }
    ).pretty()

on MongoDB Compass


If the messages are stored in a xml format, we could use the Beautiful Soup library to scrap the data:

    db = BeautifulSoup(open('db.xml').read(), "lxml")
    messages = db.findAll('message')

If the messages are stored on a txt file, we could scrap then using something as simple as:

    messages = []
    with open(db.txt, "rb") as incoming:
            for line in incoming:
                if line.startswith('user'):
                    messages.append(line)

## Getting the messages from the database

For this very particular notebook, we are going to use a very small database containing
some conversation between costumers and the attendants. The costumers are identified by the id "user_1"


# Pre-processing

In order to get the messages classified by topics, we must perform some pre-processing on them.

The LDA Topic Modeling sees the documents as bag of words (BOW),
so we need to start by transforming each message that way.

The first step to get our BOW, we must build a token generator that does:

- lowercase on each word
- remove numbers, we are assuming here that they won't help here
- remove small words, with less than 2 characters long
- spell check
- stem the words (IE get only their "root")

Those restrictions are going to be applied in order to avoid unnecessary complexity: there's no evident gains otherwise.

So, we are assuming the following words to be seen as the same by our model:
- Mobile|mobile|mobiles
- boy|boys
- girl|girls
- samsung|sansumg|Sannsungui



The chosen spell checker is the *enchant project*
It depends on the local dictionary, so we must install myspell:


Além disso, vamos adicinar algumas palabras para o dicionário, como por ex. marcas e produtos:


Besides, lets add some words to the dictionary, as manufactures and products:


### Getting better suggestions from enchant

The enchant.suggest() method provides a list of candidates for fixing the spelling.
Usually the first option is the best, but it doesn't always works as expected:



Por isso, devemos escolher a correção apropriada através de uma comparação de similaridade.

Instead, we should chose a proper fix through a similarity comparison:

Thus, we will use the methods:

    difflib.SequenceMatcher()
    difflib.SequenceMatcher().ratio()

The sequenceMatcher method compares pairs in a human friendly way.
The ratio() method evaluates the similarity of the pair. Values above 0.6 indicates we have a match.

O método sequenceMatcher compara pares de sequências de uma forma "human friendly". O método ratio() mede a similaridade do par comparado. Valores acima de 0.6 indicam um match.



## The token generator


The token generator will transform the messages into BOW.

It must:

    - lowercase on each word
    - split the message into words, removing numbers
    - remove small words, with less than 2 characters long
    - apply spell check
    - stem the words

Stemming is faster but lemmatizing is more precise, although it takes more computing time

From the wikipedia

    *Lemmatisation is closely related to stemming. The difference is that a stemmer operates on a single word without knowledge of the context, and therefore cannot discriminate between words which have different meanings depending on part of speech. However, stemmers are typically easier to implement and run faster, and the reduced accuracy may not matter for some applications.*

http://en.wikipedia.org/wiki/Lemmatisation


The choice between the former or the later really depends on the application. Sometimes just stemming will be enough.

### Applying the token generator

Lets now transform the messages from our database in tokens, as this is required to later obtain our bag of words.


## Applying the LDA


The *gensin* package brings the tools needed to implement a LDA analysis in python

https://radimrehurek.com/gensim/models/ldamodel.html


## Training our LDA model

We must choose a starting number of topics according to our knowledge about the collection, just as we would do when performing a k-mens clustering, where we have to choose a starting number of clusters. Later we will analyze the proper metrics and decide whether to increase or decrease the number of clusters|topics.

Setting the parameters for the lda training:

- lets start assuming our collection contains 5 different topics.
- lets run 100 passes over the collection until it reaches convergence about topic separation.
- the alpha parameter its about the document-topic density. A higher value indicates that each document contains more topics. We expect the messages from our costumers to be about one topic only, so we should use a small number here.


Lets also record the time it takes to finish the process.


Visualizando os 3 principais termos de cada tópico


Serializing the model

So we don't have to train the model again - lets save it to disk.
Para não ter que treinar novamente (salvando o modelo para o disco)


Análise visual  dos tópicos obtidos

Visual analysis of the topics

The bigger the distinction between groups, the better.
We can improve this distinction by adjusting the model parameters when training it.

Showing the identified topics


# modelo_salvo[mapa] irá retornar todos os tópicos possíveis e uma
# pontuação que informa a probabilidade de qua a mensagem pertençaàquele tópico





## Labeling the topics


After applying the model, we can now look at a small sample of each group so we can add a friendly label do it.

Applying the model to new messages


Keyword detection

The keywords, together with the topic prediction, will help to compose a helpful answer.


 ## Order Number

Device type


# Composing the answer

The final answer to the user message will be composed according to:

- the predicted topic - each topic will have an auxiliary function
- detected keywords


We will make use of the following auxiliary functions:

- get_topic() - ok
- get_device() - ok
- get_order() - ok
- order_status()
- salute_back()
- unknown_msg()


 afternoon


 ["All ok here on Earth. How can I help you?",
                "Please be more specific...",
                "I'm just a tired robot. Please explain, slowly...",
                "is this a new kind of joke?"]




Lets now create a dictionary containing answers to each topic.
The first answer on each class assumes that there are no keywords on the message.
The second message on each class assumes that keywords are present.


m_delivery = ["Our delivery time is 5 working days. Please inform the order number so I can fetch more information on it",

"Our delivery time is 5 working days. Lets check the order status {}."]



m_request = ["Sure thing! What's the order number?",
                 "Ok. Lets check the order status {}."]

m_indication = ["You want indications for what type of device?",
"Here are the best deals for {}."



m_comparisson = ["We can help you to choose the best options. What kind of device are you looking for?",
                "These are the best options for {}."]

m_problem= ["Easy! Everything can be fixed. What's going on?",
              "Easy! Lets fix this issue with {} the best we can."]



Answering the costumer

Now that all the auxiliary functions are ready, we can compose our answer:


# Scalability

In order to optimize the processing time, the *gensim* package offers alternate ways to train a LDA model.

Among them, are:

- set up a cluster and compute the job in a distributed fashion.
- Instead of running 100 or n steps, the batch mode, we can use the "on line" mode, where only a subset of size m of the messages will be taken into account to train the model. After the model is ready, it will take another subset, process it and then update the model. It will keep processing the subsets and updating the final model.
- a mix of both modes


Next, we will compare both modes in terms of computation time.

To do so, lets simulate a collection 1000 times larger than ours.



It was necessary 50 seconds to process the model in batch mode on this machine (Core i5, 4Gb of RAM)

unsupervised training process
