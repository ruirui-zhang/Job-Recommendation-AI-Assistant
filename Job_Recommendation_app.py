import requests
import openai
import streamlit as st
from streamlit_pills import pills
import numpy as np
import bz2
import pandas as pd


# openai.organization = ORG_KEY

openai.api_key = YOUR_KEY

st.subheader("AI Job haunt Assistant : Streamlit + OpenAI")
selected = pills("", ["Chatbot", "Similarity"], ["🎈", "🌈"])

user_input = st.text_input("You: ",placeholder = "Ask me anything ...", key="input")
with bz2.BZ2File('Data/one_per_new_df.bz2', 'r') as f:
    # read the DataFrame from the compressed file
    new_df = pd.read_csv(f, compression='bz2')
# # Load the new_df dataframe from the pickle file
# with open("Data/new_df.pkl", "rb") as f:
#     new_df = pickle.load(f)

    

new_df=new_df.rename(columns = {'json.schemaOrg.title':'Title','text':'Job Description','json.schemaOrg.jobLocation.address.addressLocality':'Location'})
job_titles = new_df["Title"].values
job_descriptions = new_df["Job Description"].values

def return_keywords():
    common_skills = [
    "Python",
    "R",
    "SQL",
    "Java",
    "Scala",
    "Git",
    "Big Data",
    "Hadoop",
    "Spark",
    "Data Visualization",
    "Tableau",
    "Power BI",
    "Data Analysis",
    "Pandas",
    "NumPy",
    "Excel",
    "Statistics",
    "Probability",
    "Machine Learning",
    "Deep Learning",
    "ETL",
    "Data Warehousing",
    "Databases",
    "MySQL",
    "PostgreSQL",
    "Oracle",
    "MongoDB",
    "Cassandra",
    "Redis",
    "AWS",
    "Google Cloud Platform",
    "Microsoft Azure",
    "Agile",
    "Scrum",
    "Kanban",
    ]

    data_scientist_skills = [
        "Scikit-learn",
        "TensorFlow",
        "Keras",
        "PyTorch",
        "Seaborn",
        "Matplotlib",
        "ggplot2",
        "Linear Algebra",
        "Calculus",
        "Optimization",
        "Modeling",
        "Feature Engineering",
        "Natural Language Processing",
        "Computer Vision",
        "Time Series Analysis",
    ]

    machine_learning_engineer_skills = [
        "Scikit-learn",
        "TensorFlow",
        "Keras",
        "PyTorch",
        "Reinforcement Learning",
        "Unsupervised Learning",
        "Supervised Learning",
        "Semi-supervised Learning",
        "Anomaly Detection",
        "Model Deployment",
        "Docker",
        "Kubernetes",
    ]

    data_analyst_skills = [
        "Data Cleaning",
        "Data Manipulation",
        "Data Exploration",
        "Statistical Analysis",
        "A/B Testing",
        "Regression Analysis",
        "Hypothesis Testing",
        "SQL Queries",
    ]

    software_engineer_skills = [
        "C++",
        "C#",
        "JavaScript",
        "TypeScript",
        "HTML",
        "CSS",
        "REST",
        "GraphQL",
        "Web Development",
        "Mobile Development",
        "Desktop Development",
        "Distributed Systems",
        "Microservices",
        "CI/CD",
        "Jenkins",
        "Travis CI",
        "Azure DevOps",
    ]

    data_engineer_skills = [
        "Hadoop",
        "Spark",
        "Kafka",
        "Hive",
        "Pig",
        "Data Pipeline",
        "API",
        "Airflow",
        "Luigi",
        "DBT",
        "Algorithms",
        "Data Structures",
    ]

    # Combine all the skill lists
    all_skills = common_skills + data_scientist_skills + machine_learning_engineer_skills + data_analyst_skills + software_engineer_skills + data_engineer_skills

    # Deduplicate the list
    all_skills = list(set(all_skills))
    return all_skills


def get_including_keywords_list(text):
    all_list = return_keywords()
    words = text.split(' ')
    keyword_list = {}
    for word in words:
        if word in all_list:
            keyword_list.append(word)
    return keyword_list
    

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


if st.button("Submit", type="primary"):
    st.markdown("----")
    res_box = st.empty()
    if selected == "Similarity":
        report = []
        for job_desc in job_descriptions:


            # MODEL = 'text-similarity-davinci-001'
            model = 'text-embedding-ada-002'
            resp = openai.Embedding.create(
                input=[user_input, job_desc],
                engine=model)

            embedding_a = resp['data'][0]['embedding']
            embedding_b = resp['data'][1]['embedding']

            similarity_score = np.dot(embedding_a, embedding_b)
            result = f"similarity scores is {similarity_score}"
            report.append(similarity_score)
        top_recommendations = new_df[['Title', 'Location', 'Job Description']].copy()
        top_recommendations['similarity_score'] = report.reshape(-1)
        top_recommendations = top_recommendations.sort_values('similarity_score', ascending=False).head(5)

        # Display top 5 job recommendations
        st.write("Top 5 job recommendations:")
        st.write(top_recommendations[['Title', 'Location','Job Description']])


        # res_box.markdown(f'*{result}*') 
            
    else:
        completions = openai.Completion.create(model='text-davinci-003',
                                            prompt=user_input,
                                            max_tokens=120, 
                                            temperature = 0.5,
                                            stream = False)
        result = completions.choices[0].text
        
        res_box.write(result)
st.markdown("----")


# # Job matching
# job_scores = []
# for job in jobs:
#     common_skills = set(user_skills).intersection(set(job["skills"]))
#     score = len(common_skills)
    
#     # Use GPT-4 to get additional insights (e.g., relevance score)
#     prompt = f"How relevant is a job requiring {', '.join(job['skills'])} for a person with skills in {', '.join(user_skills)}?"
#     relevance_text = gpt4_api_request(prompt)
#     # Parse the relevance_text to get a score (e.g., based on a scale of 1-10)
#     relevance_score = parse_relevance_text(relevance_text)
    
#     total_score = score + relevance_score
#     job_scores.append((job["title"], total_score))

# # Sort jobs by total_score (higher score = better match)
# job_scores.sort(key=lambda x: x[1], reverse=True)

# # Display the sorted job recommendations
# for job_title, score in job_scores:
#     print(f"{job_title}: {score}")
