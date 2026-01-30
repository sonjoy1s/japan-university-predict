import gradio as gr 
import pandas as pd
import numpy as np
import pickle



# Load Pickel File :

with open("japan_university_results.pkl","rb") as file:
    model = pickle.load(file)



def predict_ranking(
        University_Name, Founded_Year, Institution_Type,
        Region, Research_Impact_Score, Intl_Student_Ratio
):
    input_data = pd.DataFrame({
        "University_Name": [University_Name],
        "Founded_Year": [Founded_Year],
        "Institution_Type": [Institution_Type],
        "Region": [Region],
        "Research_Impact_Score": [Research_Impact_Score],
        "Intl_Student_Ratio": [Intl_Student_Ratio]
    })
    
    prediction = model.predict(input_data)
    return f"Predicted University Ranking: {int(prediction[0])}"


# Gradio Interface
interface = gr.Interface(
    fn=predict_ranking,
    inputs=[
        gr.Textbox(label="University Name"),
        gr.Number(label="Founded Year"),
        gr.Dropdown(choices=["Public", "Private"], label="Institution Type"),
        gr.Dropdown(choices=["Kanto", "Kansai", "Chubu", "Tohoku", "Kyushu", "Hokkaido", "Chugoku", "Shikoku"], label="Region"),
        gr.Number(label="Research Impact Score"),
        gr.Number(label="International Student Ratio")
    ],
    outputs=gr.Textbox(label="Predicted University Ranking"),
    title="Japan University Ranking Predictor",
    description="Predict the ranking of Japanese universities based on various features."
)

app = gr.Interface(
    fn=predict_ranking,
    inputs=[
        gr.Textbox(label="University Name"),
        gr.Number(label="Founded Year"),
        gr.Dropdown(choices=["Public", "Private"], label="Institution Type"),
        gr.Dropdown(choices=["Kanto", "Kansai", "Chubu", "Tohoku", "Kyushu", "Hokkaido", "Chugoku", "Shikoku"], label="Region"),
        gr.Number(label="Research Impact Score"),
        gr.Number(label="International Student Ratio")
    ],
    outputs=gr.Textbox(label="Predicted University Ranking"),
    title="Japan University Ranking Predictor",
    description="Predict the ranking of Japanese universities based on various features."
)

app.launch()

