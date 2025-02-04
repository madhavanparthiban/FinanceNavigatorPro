import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from database import database  # Import database from external file
import random

# Cache model loading
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load model
model = load_model()

# Function to find the best match
def find_best_match_bert(query, database):
    questions = [entry[0] for entry in database]
    query_embedding = model.encode([query])
    question_embeddings = model.encode(questions)
    cosine_sim = cosine_similarity(query_embedding, question_embeddings)
    best_match_index = cosine_sim.argmax()
    return database[best_match_index][1]

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize session state for quiz
if "quiz_started" not in st.session_state:
    st.session_state.quiz_started = False
    st.session_state.quiz_score = 0
    st.session_state.question_index = 0

# Define a set of quiz questions and answers
quiz_questions = [
    {"question": "What is an emergency fund?", "options": ["A savings account", "A fund for unplanned expenses", "A retirement plan"], "answer": "A fund for unplanned expenses"},
    {"question": "How much should you save for an emergency fund?", "options": ["1-2 months of expenses", "3-6 months of expenses", "12 months of expenses"], "answer": "3-6 months of expenses"},
    {"question": "Why is budgeting important?", "options": ["To track your spending", "To reduce your savings", "To increase debt"], "answer": "To track your spending"},
    {"question": "What is a 401(k)?", "options": ["A type of loan", "A retirement savings plan", "A credit card"], "answer": "A retirement savings plan"},
    {"question": "What is compound interest?", "options": ["Interest on the principal only", "Interest on both the principal and previously earned interest", "A type of loan"], "answer": "Interest on both the principal and previously earned interest"},
    {"question": "What is a credit score?", "options": ["A measure of your financial health", "A measure of your income", "A measure of your debt"], "answer": "A measure of your financial health"},
    {"question": "What is the purpose of a budget?", "options": ["To limit your spending", "To track your income and expenses", "To increase your debt"], "answer": "To track your income and expenses"},
    {"question": "What is a Roth IRA?", "options": ["A type of loan", "A retirement account with tax-free growth", "A savings account"], "answer": "A retirement account with tax-free growth"},
    {"question": "What is the 50/30/20 rule?", "options": ["A budgeting rule", "A tax rule", "A loan repayment rule"], "answer": "A budgeting rule"},
    {"question": "What is a mutual fund?", "options": ["A type of insurance", "A pool of money from many investors", "A type of loan"], "answer": "A pool of money from many investors"},
    {"question": "What is a stock?", "options": ["A type of bond", "A share in the ownership of a company", "A type of loan"], "answer": "A share in the ownership of a company"},
    {"question": "What is a bond?", "options": ["A type of stock", "A loan to a company or government", "A type of savings account"], "answer": "A loan to a company or government"},
    {"question": "What is a credit card?", "options": ["A type of loan", "A card that allows you to borrow money up to a certain limit", "A type of savings account"], "answer": "A card that allows you to borrow money up to a certain limit"},
    {"question": "What is a debit card?", "options": ["A card that deducts money directly from your bank account", "A type of loan", "A type of credit card"], "answer": "A card that deducts money directly from your bank account"},
    {"question": "What is a mortgage?", "options": ["A type of credit card", "A loan used to buy real estate", "A type of savings account"], "answer": "A loan used to buy real estate"},
    {"question": "What is a credit report?", "options": ["A report of your income", "A report of your credit history", "A report of your savings"], "answer": "A report of your credit history"},
    {"question": "What is a tax deduction?", "options": ["A reduction in your income", "A reduction in your taxable income", "A type of loan"], "answer": "A reduction in your taxable income"},
    {"question": "What is a tax credit?", "options": ["A credit on your tax bill", "A type of loan", "A type of savings account"], "answer": "A credit on your tax bill"},
    {"question": "What is a financial planner?", "options": ["A person who helps you manage your finances", "A type of loan", "A type of savings account"], "answer": "A person who helps you manage your finances"},
    {"question": "What is a certificate of deposit (CD)?", "options": ["A type of loan", "A savings account with a fixed term and interest rate", "A type of credit card"], "answer": "A savings account with a fixed term and interest rate"},
]

# Function to handle quiz logic
def start_quiz():
    if not st.session_state.quiz_started:
        st.session_state.quiz_started = True
        st.session_state.quiz_score = 0
        st.session_state.question_index = 0
        
        # Shuffle and pick 5 random questions
        random.shuffle(quiz_questions)
        st.session_state.selected_questions = quiz_questions[:5]  # Select only the first 5 shuffled questions

    # Get current question from selected random questions
    question = st.session_state.selected_questions[st.session_state.question_index]

    # Display the question and options
    st.write(f"**Question {st.session_state.question_index + 1}:** {question['question']}")
    user_answer = st.radio("Select an answer:", question["options"], key=f"question_{st.session_state.question_index}")

    # Proceed to next question or show the score
    if st.button("Next"):
        # Check if the selected answer is correct
        if user_answer == question["answer"]:
            st.session_state.quiz_score += 1
        st.session_state.question_index += 1

        # If there are more questions, rerun to show next question
        if st.session_state.question_index < len(st.session_state.selected_questions):
            st.rerun()  # Refresh the page to show next question
        else:
            st.write(f"**Quiz Finished!** Your score is: {st.session_state.quiz_score}/{len(st.session_state.selected_questions)}")
            st.session_state.quiz_started = False  # Reset quiz state after finishing

# Streamlit Sidebar for Navigation
tabs = ["Chatbot", "Quiz"]
selected_tab = st.sidebar.selectbox("Select a Tab", tabs)

# Content based on selected tab
if selected_tab == "Chatbot":
    st.title("💰 Personal Finance Chatbot")
    st.write("Ask me anything about your personal finances!")

    # Display chat history
    for msg in st.session_state.chat_history:
        st.write(f"**You:** {msg['query']}")
        st.success(f"💬 {msg['response']}")

    # Use a form to make "Ask" button the default action
    with st.form(key="chat_form"):
        query = st.text_input("Enter your question:", key=f"chat_input_{len(st.session_state.chat_history)}")
        submit_button = st.form_submit_button("Ask")

    if submit_button and query:
        response = find_best_match_bert(query, database)

        # Store the query and response in chat history
        st.session_state.chat_history.append({"query": query, "response": response})

        # Rerun to update the chat history display
        st.rerun()

elif selected_tab == "Quiz":
    start_quiz()