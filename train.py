# Step 0: Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# --- Step 1: Load your dataset ---
# IMPORTANT: This is the full path to YOUR file.
# The 'r' before the string is crucial for Windows paths.
file_path = r'E:\College\Projects\IBM DFL\code\ML Model 2\Scam_Awareness___Prevention_Q_A_Dataset_NoRepeatedQuestions.csv'

try:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    print("File loaded successfully!")
except FileNotFoundError:
    # If the file is not found, print a helpful error message and exit.
    print(f"Error: The file was not found at the path: {file_path}")
    print("Please make sure the file path is correct and the file exists in that location.")
    exit() # Stop the script if we can't find the file.

# --- Step 2: Prepare the data and train the model ---
print("Training model...")

# Select the column containing the questions
X = df['user_question']

# Create a TF-IDF Vectorizer to convert text questions into numerical vectors.
# 'stop_words='english'' helps ignore common English words like 'the', 'a', 'is'.
vectorizer = TfidfVectorizer(stop_words='english')

# Fit the vectorizer on your questions and transform them into a matrix of numbers.
X_vectorized = vectorizer.fit_transform(X)

# Create the K-Nearest Neighbors model.
# 'n_neighbors=1' means we want to find the single closest match for any new question.
# 'metric='cosine'' is a good choice for measuring similarity between text vectors.
knn = NearestNeighbors(n_neighbors=1, metric='cosine')

# Fit the KNN model on the vectorized (numerical) question data.
knn.fit(X_vectorized)

print("Model training complete!")


# --- Step 3: Create the function that finds and returns an answer ---
def get_answer(question):
    """
    This function takes a new question from the user, finds the most similar
    question in the dataset, and returns the corresponding detailed answer.
    """
    # 1. Transform the user's new question into a numerical vector using the SAME vectorizer.
    question_vectorized = vectorizer.transform([question])
    
    # 2. Use the trained KNN model to find the nearest neighbor to the user's question.
    # It returns the distance and the index of the neighbor.
    distances, indices = knn.kneighbors(question_vectorized)
    
    # 3. Get the index of the best match (it's the first one in the list).
    best_match_index = indices[0][0]
    
    # 4. Use that index to look up and return the 'detailed_answer' from the original DataFrame.
    return df['detailed_answer'].iloc[best_match_index]


# --- Step 4: Create an interactive loop to chat with the user ---
print("\n--- Scam Awareness Chatbot ---")
print("You can now ask questions about financial scams.")
print("Type 'quit' or 'exit' to stop the program.")

# This loop will run forever until the user types 'quit' or 'exit'.
while True:
    # Prompt the user to enter their question.
    user_input = input("\nYou: ")

    # Check if the user wants to end the chat.
    if user_input.lower() in ['quit', 'exit']:
        print("Chatbot: Goodbye! Stay safe.")
        break # Exit the loop.

    # If the user asks a question, call the get_answer function.
    answer = get_answer(user_input)
    
    # Print the answer for the user.
    print(f"Chatbot: {answer}")