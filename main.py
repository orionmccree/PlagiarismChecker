import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Creates a list of .txt files in the root folder, then cleans them for efficient usage
def load_text_files():
    text_files_list = [doc for doc in os.listdir() if doc.endswith('.txt')]
    text_file_contents = [open(text_file, encoding='utf-8').read() for text_file in text_files_list]
    return text_files_list, text_file_contents

# Converts a list of .txt files into a TF-IDF vector representation
def transform_to_tfid(text_data):
    return TfidfVectorizer().fit_transform(text_data).toarray()

# Computers the cosine similarity score to efficiently calculate the similarity between two documents based on their TF-IDF vectors
def compute_similarity(vector1, vector2):
    return cosine_similarity([vector1, vector2])[0][1]

# Calculates the simimlarity between all pairs of .txt files, then formats and returns the printed results
def calculate_plagiarism():
    file_names, file_contents = load_text_files()
    vector_text = transform_to_tfid(file_contents)
    vector_list = list(zip(file_names, vector_text))
    plagiarism_checker_results = set()

    # Goes through both text files and creates a score based on the similarity of the vectors
    for author1, text_vector1 in vector_list:
        new_vectors = vector_list.copy()
        current_index = new_vectors.index((author1, text_vector1))
        del new_vectors[current_index]
        
        for author2, text_vector2 in new_vectors:
            similarity_score_percentage = compute_similarity(text_vector1, text_vector2)
            # Round the similarity score then convert to a percentage
            similarity_score_percentage = str((round(similarity_score_percentage, 4) * 100)) + '%'
             
            sorted_authors_list = sorted((author1, author2))
            similarity_score = 'The similarity score between ' + str(sorted_authors_list[0]) + ' and ' + str(sorted_authors_list[1]) + 'is: ' + str(similarity_score_percentage)
            plagiarism_checker_results.add(similarity_score)

    #  Returns the overall printed results
    return plagiarism_checker_results

# Entry point of the program to print plagiarism results
if __name__ == "__main__":
    for data in calculate_plagiarism():
        print(data)
