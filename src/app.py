import gradio as gr

def predict(text):
    return f"You typed: {text}"

demo = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text"
)

demo.launch()