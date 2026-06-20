import gradio as gr

print("APP.PY LOADED")

def greet(name):
    return f"Hello {name}"

demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(),
    outputs=gr.Textbox(),
    title="Test Space"
)