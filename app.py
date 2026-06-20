import gradio as gr

print("ROOT APP LOADED")

def greet(name):
    return f"Hello {name}"

demo = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text"
)