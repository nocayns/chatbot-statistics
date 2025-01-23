import os
import sys
import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from crewai import Agent, Task, Crew
import pysqlite3
import replicate

# Replace sqlite3 module with pysqlite3 for compatibility
sys.modules["sqlite3"] = pysqlite3

# Set up API key 
os.getenv('REPLICATE_API_TOKEN')

# Streamlit app title
st.title('🧮 Demo Chatbot: Math Assistant')

# User input field
input_text = st.text_input("Enter a math question or topic:")

# Memory for conversation history
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,  # Keeps the last `memory_size` interactions in memory
    return_messages=True  # Ensures that the messages are returned
)

# Function to set up and get response from agents
def get_response(question):
    # Initialize LlamaAPI
    llm = LlamaAPI(api_key=api_key)

    # Define Professor Agent
    professor = Agent(
        role="Math Professor",
        goal=("Memberikan solusi kepada para siswa yang bertanya tentang "
              "pertanyaan matematika dan memberi mereka jawaban."),
        backstory=("Anda adalah seorang profesor matematika yang bisa menyelesaikan "
                   "pertanyaan matematika dengan cara yang dapat dipahami semua orang."),
        allow_delegation=False,
        verbose=True,
        llm=llm
    )

    # Define Reviewer Agent
    reviewer = Agent(
        role="Reviewer",
        goal=("Memeriksa dan mengoreksi jawaban yang diberikan oleh profesor "
              "untuk memastikan keakuratannya, lalu memberikan jawaban yang benar setelah dikoreksi."),
        backstory=("Anda adalah seorang ahli matematika yang teliti yang bertugas "
                   "memverifikasi dan mengoreksi jawaban untuk memastikan bahwa siswa menerima jawaban yang benar."),
        allow_delegation=False,
        verbose=True,
        llm=llm
    )

    # Define tasks for agents
    task_for_professor = Task(
        description=f"Berikan solusi untuk pertanyaan berikut: {question}",
        expected_output="Berikan jawaban yang jelas dan benar untuk pertanyaan.",
        agent=professor
    )

    task_for_reviewer = Task(
        description=f"Periksa jawaban dari profesor untuk pertanyaan: {question}.",
        expected_output="Hanya berikan jawaban yang telah dikoreksi, tanpa komentar tambahan.",
        agent=reviewer
    )

    # Create crew with both agents
    crew = Crew(
        agents=[professor, reviewer],
        tasks=[task_for_professor, task_for_reviewer],
        verbose=True
    )

    # Execute the tasks and get the result
    result = crew.kickoff()

    # Return the result from the crew
    return result

# Function to generate response based on the user's question
def generate_response(question):
    try:
        response = get_response(question)
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

# Handle submit logic
def handle_submit(question):
    with st.spinner('Mohon menunggu sebentar...'):
        response = generate_response(question)
        if response:
            st.success("Jawaban telah dihasilkan:")
            st.write(response)

# Submit button logic
if st.button("Submit"):
    if input_text:
        handle_submit(input_text)
    else:
        st.warning("Please enter a question before submitting.")
