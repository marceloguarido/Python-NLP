# model[map] will return all the possible topics and a score that tells
# the probability of the message belonging to that topic.
#
# we will order it to get the topic which has the biggest probability.
# Also, if the probability is lower than 60%, we will chose to say that we just
# don't know what the message is about (category 6)


# vizualisando uma fração das mensagens de cada grupo, para ajudar a definir o
título de cada tópico
# foi usado 80% porque nosso db é muuuuito pequeno

# showing a sample of each group, to aid labeling each one
