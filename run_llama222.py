from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import streamlit as st
import os

# Set API Key, for now, it is set to "NA" since it's not used
os.environ["OPENAI_API_KEY"] = "NA"

# Streamlit app title
st.title('Demo Chatbot')

# User input field
input_text = st.text_input("Enter a math question or topic")

# Memory for conversation history
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,  # Keeps the last 5 interactions in memory
    return_messages=True  # Ensures that the messages are returned
)

# Function to set up and get response from agents
def get_response(question):
    # Initialize LLM from ChatOpenAI
    llm = ChatOpenAI(
        model="crewai-llama2",
        base_url="http://localhost:11434/v1"
    )

    # Define Professor Agent
    professor = Agent(
        role="Math Professor",
        goal="Memberikan solusi kepada para siswa yang bertanya tentang pertanyaan matematika dan memberi mereka jawaban.",
        backstory="Anda adalah seorang profesor matematika yang bisa menyelesaikan pertanyaan matematika dengan cara yang dapat dipahami semua orang.",
        allow_delegation=False,
        verbose=True,
        llm=llm
    )

    # Define Reviewer Agent
    reviewer = Agent(
        role="Reviewer",
        goal="Memeriksa dan mengoreksi jawaban yang diberikan oleh profesor untuk memastikan keakuratannya, lalu memberikan jawaban yang benar setelah dikoreksi.",
        backstory="Anda adalah seorang ahli matematika yang teliti yang bertugas memverifikasi dan mengoreksi jawaban untuk memastikan bahwa siswa menerima jawaban yang benar.",
        allow_delegation=False,
        verbose=True,
        llm=llm
    )

    # Define the task for the Math Professor Agent
    task_for_professor = Task(
        description=f"Berikan solusi untuk pertanyaan berikut: {question}",
        expected_output="Berikan jawaban yang jelas dan benar untuk pertanyaan.",
        agent=professor
    )

    # Define the task for the Reviewer Agent
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
    response = get_response(question)
    return response

# Handle submit logic
def handle_submit(response):
    with st.spinner('Mohon menunggu sebentar...'):
        # Generate the response
        response = generate_response(response)
        # Display the result
        st.write(response)

# Submit button logic
if st.button("Submit"):
    if input_text:
        handle_submit(input_text)  
    else:
        st.warning("Please enter a question before submitting.")