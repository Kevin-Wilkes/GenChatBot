import nltk
from nltk.corpus import wordnet as wn

# Download WordNet if needed
nltk.download('wordnet')

def build_person_nouns():
    person_nouns = set()
    # 'person.n.01' is the WordNet synset for "person"
    person_synset = wn.synset('person.n.01')

    # Get all hyponyms (i.e., specific types of persons)
    for hyponym in person_synset.closure(lambda s: s.hyponyms()):
        for lemma in hyponym.lemma_names():
            # Add clean names (no underscores)
            person_nouns.add(lemma.replace("_", " ").lower())

    return person_nouns

PERSON_NOUNS = build_person_nouns()

# Example
print(list(PERSON_NOUNS)[:20])  # show 20 sample person nouns

with open('./Filter_Data/Person_Nouns.txt', 'w') as file:
    file.write(str(PERSON_NOUNS))
