# app/bigram_model.py

from collections import defaultdict
import random
import spacy # Import the spacy library
import numpy as np # Import numpy for vector operations

class BigramModel:
    def __init__(self, corpus):
        """
        Initializes the BigramModel.
        - Loads the spaCy 'en_core_web_lg' model.
        - Builds the bigram dictionary from the corpus.
        """
        # Load the large English language model from spaCy
        print("Loading spaCy model 'en_core_web_lg'...")
        self.nlp = spacy.load("en_core_web_lg")
        print("Model loaded.")
        
        self.bigrams = defaultdict(list)
        self.build_bigrams(corpus)
    
    def build_bigrams(self, corpus):
        """
        Builds a dictionary of bigrams from the corpus.
        This functionality remains the same.
        """
        for sentence in corpus:
            words = sentence.split()
            for i in range(len(words)-1):
                self.bigrams[words[i]].append(words[i+1])
    
    def generate_text(self, start_word, length):
        """
        Generates text using the original bigram logic.
        This functionality remains the same.
        """
        word = start_word
        result = [word]
        for _ in range(length-1):
            next_words = self.bigrams.get(word)
            if not next_words:
                break
            word = random.choice(next_words)
            result.append(word)
        return ' '.join(result)

    ## --- New spaCy-powered Methods --- ##

    def get_word_embedding(self, word):
        """
        Calculates the word embedding (vector) for a given word.
        
        Args:
            word (str): The word to get the embedding for.
            
        Returns:
            numpy.ndarray: The word embedding vector.
        """
        return self.nlp(word).vector

    def get_similarity(self, text1, text2):
        """
        Calculates the semantic similarity between two texts (words or sentences).
        Similarity is a float between 0 (not similar) and 1 (very similar).
        
        Args:
            text1 (str): The first word or sentence.
            text2 (str): The second word or sentence.
            
        Returns:
            float: The cosine similarity between the two texts.
        """
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        return doc1.similarity(doc2)

# --- Example Usage ---
if __name__ == '__main__':
    # 1. Define a sample corpus
    corpus = [
        "the king is powerful",
        "the queen is wise",
        "the king rules the land",
        "the queen leads the people",
        "a cat sat on the mat",
        "a dog chased the cat"
    ]

    # 2. Create an instance of the model
    # This will automatically load the spaCy model
    model = BigramModel(corpus)
    
    print("\\n--- Original Bigram Functionality ---")
    # 3. Use the original text generation method
    generated_text = model.generate_text("the", 5)
    print(f"Generated Text: '{generated_text}'")

    print("\\n--- New Word Embedding Functionality ---")
    
    # 4. Get the word embedding for 'king'
    king_vector = model.get_word_embedding("king")
    print(f"Embedding vector for 'king' (first 5 elements): {king_vector[:5]}")
    
    # 5. Calculate similarity between words
    similarity_king_queen = model.get_similarity("king", "queen")
    similarity_king_cat = model.get_similarity("king", "cat")
    print(f"Similarity between 'king' and 'queen': {similarity_king_queen:.4f}  👑")
    print(f"Similarity between 'king' and 'cat':   {similarity_king_cat:.4f}  🐱")
    
    # 6. Calculate similarity between sentences
    query = "who governs the country?"
    info1 = "the king rules the land"
    info2 = "the dog chased the cat"
    
    similarity_sent1 = model.get_similarity(query, info1)
    similarity_sent2 = model.get_similarity(query, info2)
    print(f"\\nSimilarity between '{query}' and '{info1}': {similarity_sent1:.4f}")
    print(f"Similarity between '{query}' and '{info2}': {similarity_sent2:.4f}")