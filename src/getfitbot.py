import json
import numpy as np
import pandas as pd
import gradio as gr
import pickle as pkl
from ludwig.api import LudwigModel

model = LudwigModel.load("trained")


def predict(inputs):
    df = inputs.dropna()
    Yhat = model.predict(df)[0]
    return(Yhat)


iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.inputs.Dataframe(
            headers=["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"],
            datatype=["str", "number", "number", "number", "number", "number", "number"]
        ),
    ],
    outputs=gr.outputs.Dataframe(headers=["Predicted calories"]),
)
if __name__ == "__main__":
    app, local_url, share_url = iface.launch()
