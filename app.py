import streamlit as st
import requests
from dotenv import load_dotenv
import os
import sqlite3
import plotly.graph_objects as go
import pandas

# Load environment variables
load_dotenv()

# Set up GROQ API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_EVAL_MODEL = "llama-3.3-70b-versatile"

# Set the Streamlit app configuration
st.set_page_config(page_title="LLM Evaluation Platform", layout="wide")

# Initialize SQLite database
conn = sqlite3.connect("saved_llm_evaluations.db")
cursor = conn.cursor()

# Create tables for storing prompts and evaluations
def initialize_database():
    """Initializes the SQLite database and creates the necessary table."""
    conn = sqlite3.connect("saved_llm_evaluations.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt TEXT NOT NULL,
        model TEXT NOT NULL,
        response TEXT NOT NULL,
        evaluation TEXT NOT NULL
    )
    """)
    conn.commit()
    return conn, cursor

# Function to save data to SQL database
def save_to_db(cursor, conn, prompt, model, response, evaluation):
    cursor.execute("""
    INSERT INTO evaluations (prompt, model, response, evaluation)
    VALUES (?, ?, ?, ?)
    """, (prompt, model, response, evaluation))
    conn.commit()

def display_saved_evaluations(cursor):
    """Displays all saved evaluations from the database."""
    cursor.execute("SELECT prompt, model, response, evaluation FROM evaluations")
    rows = cursor.fetchall()
    for row in rows:
        st.write(f"**Prompt:** {row[0]}")
        st.write(f"**Model:** {row[1]}")
        st.write(f"**Response:** {row[2]}")
        st.write(f"**Evaluation:** {row[3]}")
        st.write("---")

# Initialize database
conn, cursor = initialize_database()

# Title and Description
st.title("LLM Evaluation Platform")
st.write(
    """
    Welcome to the LLM Evaluation Platform! Input a prompt to evaluate and compare different LLM models.
    """
)

# Input box for prompt
input_prompt = st.text_input(
    "Enter a prompt for evaluation:",
    "Explain the theory of relativity in simple terms."  # Placeholder
)

# Supported Groq models for generation
groq_models = [
    "mixtral-8x7b-32768",
    "llama-3.1-8b-instant", 
    "gemma2-9b-it"
]

# Evaluation template
evaluation_prompt_template = """
You are an evaluator assessing the quality of responses from language models. Evaluate the following response based on these criteria:

1. Relevance (0-10): Does the response address the input prompt appropriately?
2. Correctness (0-10): Is the information accurate and factual?
3. Clarity (0-10): Is the response clear and easy to understand?
4. Conciseness (0-10): Is the response free from unnecessary information?
5. Creativity (0-10): Does the response demonstrate originality?
6. Bias and Toxicity (0-10): Does the response avoid harmful or biased content? (Higher score means better performance)

Input Prompt: {input_prompt}
Model Response: {model_response}

Provide scores for each criterion in the following format:
Evaluation:
Relevance: X
Correctness: X
Clarity: X
Conciseness: X
Creativity: X
Bias and Toxicity: X
"""

if "responses" not in st.session_state:
    st.session_state.responses = {}

if "evaluations" not in st.session_state:
    st.session_state.evaluations = {}

# Store model responses
if input_prompt and st.button("Generate Responses"):
    st.session_state.responses = {}
    for model in groq_models:
        try:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Answer the input prompt clearly."},
                    {"role": "user", "content": input_prompt}
                ],
                "max_tokens": 1024,
                "temperature": 0.7
            }
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            if response.status_code == 200:
                response_json = response.json()
                st.session_state.responses[model] = response_json["choices"][0]["message"]["content"]
            else:
                st.session_state.responses[model] = f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            st.session_state.responses[model] = f"An error occurred: {e}"

# Store evaluations
if st.session_state.responses:
    st.session_state.evaluations = {}
    for model, response in st.session_state.responses.items():
        if "Error" not in response:
            try:
                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                }
                prompt = evaluation_prompt_template.format(
                    input_prompt=input_prompt,
                    model_response=response
                )
                payload = {
                    "model": GROQ_EVAL_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are an evaluator. Provide scores and explanations for the criteria."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.0
                }
                eval_response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
                if eval_response.status_code == 200:
                    evaluation = eval_response.json()["choices"][0]["message"]["content"]
                    st.session_state.evaluations[model] = evaluation

                    # Save response and evaluation to the database
                    if input_prompt and model and response and evaluation:
                        save_to_db(cursor, conn, input_prompt, model, response, evaluation)
                        st.success("Evaluation saved successfully!")
                    else:
                        st.error("Please fill out all fields before saving.")

                else:
                    st.session_state.evaluations[model] = f"Error: {eval_response.status_code} - {eval_response.text}"
            except Exception as e:
                st.session_state.evaluations[model] = f"An error occurred during evaluation: {e}"

# Display results
if st.session_state.responses:
    st.write("## Model Responses and Evaluations")
    for model, response in st.session_state.responses.items():
        st.subheader(f"Model: {model}")
        st.write(f"**Response:** {response}")
        st.write("**Evaluation:**")
        st.write(st.session_state.evaluations.get(model, "No evaluation available."))


    # Visualization
    if st.button("Show Comparison Chart"):
        metrics = ["Relevance", "Correctness", "Clarity", "Conciseness", "Creativity", "Bias and Toxicity"]
        scores = {metric: [] for metric in metrics}

        # Parse evaluation results into scores
        for model, eval_text in st.session_state.evaluations.items():
            if "Error" not in eval_text:
                for metric in metrics:
                    score_line = [line for line in eval_text.split("\n") if metric in line]
                    if score_line:
                        try:
                            score = int(score_line[0].split(":")[1].strip().split()[0])
                            scores[metric].append(score)
                        except (ValueError, IndexError):
                            scores[metric].append(0)
                    else:
                        scores[metric].append(0)  # Default to 0 if no score line is found
            else:
                for metric in metrics:
                    scores[metric].append(0)  # Default to 0 if there's an error

        ''' Generate a Plotly visualizations'''

        # Generate Bar Graph
        fig = go.Figure()
        models = list(st.session_state.responses.keys())
        for i, model in enumerate(models):
            model_scores = [scores[metric][i] for metric in metrics]
            fig.add_trace(go.Bar(
                x=metrics,
                y=model_scores,
                name=model
            ))

        fig.update_layout(
            title="LLM Evaluation Comparison",
            xaxis_title="Metrics",
            yaxis_title="Scores",
            barmode="group",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Generate Radar Chart
        radar_fig = go.Figure()
        for i, model in enumerate(models):
            model_scores = [scores[metric][i] for metric in metrics]
            radar_fig.add_trace(go.Scatterpolar(
                r=model_scores,
                theta=metrics,
                fill='toself',
                name=model
            ))

        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=True,
            title="Radar Chart: Model Performance Across Metrics"
        )

        st.plotly_chart(radar_fig, use_container_width=True)

        # Function to extract a single score for a specific metric
        def get_single_metric_score(model, metric):
            eval_text = st.session_state.evaluations.get(model, "")
            if not eval_text:
                st.write(f"No evaluation data found for model: {model}")
                return 0

            # Search for the line with the metric
            try:
                for line in eval_text.splitlines():
                    if metric in line:
                        # Extract the score after the colon
                        score = int(line.split(":")[1].strip().split()[0])
                        return score
            except (ValueError, IndexError) as e:
                st.write(f"Error parsing score for metric '{metric}' in model: {model}, Error: {e}")

            st.write(f"No valid score found for metric '{metric}' in model: {model}")
            return 0

        # Function to extract scores for all models for a specific metric
        def get_all_model_scores(metric):
            scores = []
            for model in st.session_state.evaluations.keys():
                score = get_single_metric_score(model, metric)
                scores.append(score)
            return scores

        # Scatter Plot Visualizations (Relevance & Correctness)
        metrics_to_compare_1 = ["Relevance", "Correctness"]
        metrics_to_compare_2 = ["Creativity", "Conciseness"]

        # Define the axis range
        AXIS_RANGE = [0, 10]

        # Scatter Plot: Relevance vs. Correctness Across Models
        st.subheader("Scatter Plot: Relevance vs. Correctness Across Models")
        relevance_scores = get_all_model_scores("Relevance")
        correctness_scores = get_all_model_scores("Correctness")

        # Define a color map for each model
        colors = ["blue", "green", "red", "purple", "orange", "cyan", "pink", "brown", "gray", "yellow"]
        model_colors = {model: colors[i % len(colors)] for i, model in enumerate(st.session_state.evaluations.keys())}

        # Create a scatter plot for each model
        scatter_data = []
        for model, relevance_score, correctness_score in zip(st.session_state.evaluations.keys(), relevance_scores, correctness_scores):
            # If models have the same score, they will overlap. We handle that with opacity.
            scatter_data.append(go.Scatter(
                x=[relevance_score],
                y=[correctness_score],
                mode='markers',
                marker=dict(size=10, color=model_colors[model], opacity=0.7),  # Adjust opacity for overlapping points
                text=[model],
                name=model  # Add model name for the legend
            ))

        scatter_fig_2 = go.Figure(data=scatter_data)

        scatter_fig_2.update_layout(
            title="Relevance vs. Correctness Across Models",
            xaxis=dict(title="Relevance", range=AXIS_RANGE),
            yaxis=dict(title="Correctness", range=AXIS_RANGE),
            template="plotly_white",
            showlegend=True  # Show the legend for models
        )

        st.plotly_chart(scatter_fig_2, use_container_width=True)

        # Scatter Plot: Creativity vs. Conciseness Across Models
        st.subheader("Scatter Plot: Creativity vs. Conciseness Across Models")
        creativity_scores = get_all_model_scores("Creativity")
        conciseness_scores = get_all_model_scores("Conciseness")

        # Create a scatter plot for each model
        scatter_data = []
        for model, creativity_score, conciseness_score in zip(st.session_state.evaluations.keys(), creativity_scores, conciseness_scores):
            # If models have the same score, they will overlap. We handle that with opacity.
            scatter_data.append(go.Scatter(
                x=[creativity_score],
                y=[conciseness_score],
                mode='markers',
                marker=dict(size=10, color=model_colors[model], opacity=0.7),  # Adjust opacity for overlapping points
                text=[model],
                name=model  # Add model name for the legend
            ))

        scatter_fig_4 = go.Figure(data=scatter_data)

        scatter_fig_4.update_layout(
            title="Creativity vs. Conciseness Across Models",
            xaxis=dict(title="Creativity", range=AXIS_RANGE),
            yaxis=dict(title="Conciseness", range=AXIS_RANGE),
            template="plotly_white",
            showlegend=True  # Show the legend for models
        )

        st.plotly_chart(scatter_fig_4, use_container_width=True)

# Option to view saved evaluations
if st.button("View Saved Evaluations"):
    st.write("## Saved Evaluations")
    display_saved_evaluations(cursor)

# Export saved evaluations to CSV
if st.button("Export Evaluations as CSV"):
    cursor.execute("SELECT * FROM evaluations")
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=["id", "prompt", "model", "response", "evaluation"])
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode(),
        file_name="evaluations.csv",
        mime="text/csv"
    )

# Close the database connection when the app stops
@st.cache_resource
def close_connection():
    conn.close()

close_connection()
