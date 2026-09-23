import gradio as gr
from app.frontier_model import FrontierModelProvider

provider = FrontierModelProvider()

def decide(text):
    _, answers, _ = provider.predict(
        {'source_type': 'demo', 'body': text},
        {
            'route': {'type': 'choice', 'instructions': 'Which workstream should own this item?', 'criteria': {'safety': 'Potential safety signal.', 'quality': 'Potential product-quality issue.', 'medical_information': 'Medical-information request.', 'general': 'No supported regulated workstream.'}},
            'agent_action': {'type': 'action', 'instructions': 'Which bounded agent action is appropriate before human review?', 'criteria': {'enrich': 'Collect non-terminal evidence.', 'review': 'Place in human review.', 'escalate': 'Prioritize human review.', 'hold': 'Take no further automated action.'}},
        },
    )
    return answers

demo = gr.Interface(
    fn=decide,
    inputs=gr.Textbox(lines=8, label='State'),
    outputs=gr.JSON(label='Typed decisions'),
    title='ClinevoOne Frontier Decision Lab',
    description='Research-only System-One-style typed decisions with bounded agent actions.',
)
demo.launch()
