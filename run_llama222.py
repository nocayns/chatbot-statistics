import os
import sys
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import pysqlite3

# Replace sqlite3 module with pysqlite3 for compatibility
sys.modules["sqlite3"] = pysqlite3

# Streamlit app title
st.title('🧮 Demo Chatbot: Math Assistant')

# User input field
input_text = st.text_input("Enter a math question or topic:")

# Set Llama 2 base URL
llama2_base_url = os.getenv("LLAMA2_BASE_URL", "https://12ef-103-162-62-56.ngrok-free.app/v1")  # Replace with your actual ngrok base URL

# Check if base URL is available
if not llama2_base_url:
    st.error("LLAMA2_BASE_URL tidak ditemukan. Pastikan Anda telah mengatur base URL dengan benar.")
    sys.exit(1)


# Function to set up and get response from agents
def get_response(question):
    # Initialize Llama 2 API through CrewAI
    llm = ChatOpenAI(
        model="crewai-llama2",
        base_url=llama2_base_url,
        temperature=0.7
    )

    # Define Professor Agent
    professor = Agent(
        role="Math Professor",
        goal=(
            "Memberikan solusi kepada para siswa yang bertanya tentang pertanyaan matematika "
            "dan memberi mereka jawaban yang jelas dan mudah dimengerti."
        ),
        backstory="Anda adalah seorang profesor matematika yang andal dalam menyelesaikan masalah matematika.",
        allow_delegation=False,
        verbose=True,
        llm=llm
    )

    # Define Reviewer Agent
    reviewer = Agent(
        role="Reviewer",
        goal=(
            "Memeriksa dan mengoreksi jawaban yang diberikan oleh profesor untuk memastikan keakuratannya, "
            "lalu memberikan jawaban yang benar jika diperlukan."
        ),
        backstory="Anda adalah seorang ahli matematika yang teliti yang bertugas memverifikasi jawaban untuk memastikan akurasi.",
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
