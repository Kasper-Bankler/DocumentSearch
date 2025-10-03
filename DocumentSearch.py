import math
# Import the stemming tool
import nltk
from nltk.tokenize import word_tokenize
# Import function to find n largest items
from heapq import nlargest

# Download stemmer
nltk.download('punkt', quiet=True)


def td_idf(query_list, documents, titles):
    stemmer = nltk.stem.PorterStemmer()
    # Pre initialize list
    tf_idf_list = [0.0]*len(documents)

    for query in query_list:
        number_of_documents_with_term = 0
        query_word = stemmer.stem(query)
        # Find number of documents with term t
        for document in documents:
            if query_word in [stemmer.stem(word) for word in word_tokenize(document)]:
                number_of_documents_with_term += 1

        # Calculate TF-IDF for every document
        for i, document in enumerate(documents):
            # Tokenize and stem the text
            tokens = [stemmer.stem(word) for word in word_tokenize(document)]
            # Count the number of occurences of the query word
            token_count = tokens.count(query_word) / len(tokens)
            # Calculate TF-IDF
            tf_idf = token_count * \
                math.log(len(documents)/number_of_documents_with_term)
            # Add TF-IDF to list
            tf_idf_list[i] += tf_idf

    # print the top 5 items with largest term frequency
    for item_tf_idf, item_title in nlargest(5, zip(tf_idf_list, titles)):
        print(f'{item_title} (TF-IDF={item_tf_idf})')


# Open the file animals.txt and read the lines
file = open('animals.txt', encoding='utf8')
lines = file.read().splitlines()
file.close()

# Extract titles and documents
titles = lines[0::4]
documents = lines[2::4]

td_idf(['stripes', 'white'], documents, titles)
