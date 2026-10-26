
import os

import gradio as gr

from model_utils import predict_image


# Custom styling for the interface.
CUSTOM_CSS = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

#hero {
    text-align: center;
    padding: 35px 20px;
    border-radius: 22px;
    background: linear-gradient(135deg, #172554, #4338ca);
    color: white;
    margin-bottom: 22px;
}

#hero h1 {
    font-size: 38px;
    font-weight: 800;
    color: white;
}

#hero p {
    color: #e0e7ff;
    font-size: 16px;
    line-height: 1.7;
    max-width: 650px;
    margin: auto;
}

.panel {
    border: 1px solid var(--border-color-primary);
    border-radius: 18px;
    padding: 20px;
}

#footer {
    text-align: center;
    padding: 18px;
    font-size: 13px;
    color: var(--body-text-color-subdued);
}

@media (max-width: 600px) {
    #hero {
        padding: 25px 14px;
    }

    #hero h1 {
        font-size: 29px;
    }
}
"""


# Connect the uploaded image to the prediction function.
def classify_image(image):
    if image is None:
        return (
            {},
            "### Please upload an image first."
        )

    try:
        # Get the probabilities from our trained model.
        probabilities = predict_image(image)

        # Find the class with the highest probability.
        predicted_class = max(
            probabilities,
            key=probabilities.get
        )

        confidence = probabilities[predicted_class] * 100

        # Create a readable prediction summary.
        summary = f"""
### Prediction complete

**Predicted class:** {predicted_class.upper()}

**Confidence:** {confidence:.2f}%

The confidence is the model's estimated probability,
not a guarantee that the prediction is correct.
"""

        return probabilities, summary

    except (ValueError, OSError) as exc:
        return {}, f"### Prediction failed\n\n{exc}"


# Build the Gradio application.
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
    ),
    css=CUSTOM_CSS,
    title="Ants vs Bees AI Classifier",
) as demo:

    # Header banner.
    gr.HTML("""
    <div id="hero">
        <h1>Ants vs Bees AI Classifier</h1>
        <p>
            Discover what's in your image with deep learning.
            Upload a picture and let our PyTorch ResNet-18
            model predict whether it contains ants or bees.
        </p>
    </div>
    """)

    gr.Markdown(
        "### Explore image classification\n"
        "Upload a clear image of an ant or bee to begin."
    )

    with gr.Row(equal_height=False):

        # Left side: image input.
        with gr.Column(scale=1, elem_classes=["panel"]):
            gr.Markdown("## 01 · Upload image")

            image_input = gr.Image(
                label="Your insect image",
                type="pil",
                sources=["upload"],
                image_mode="RGB",
                height=340,
            )

            classify_button = gr.Button(
                "Classify image",
                variant="primary",
                size="lg",
            )

            clear_button = gr.ClearButton(
                components=[image_input],
                value="Clear image",
            )

        # Right side: prediction output.
        with gr.Column(scale=1, elem_classes=["panel"]):
            gr.Markdown("## 02 · Prediction results")

            prediction_output = gr.Label(
                label="Class probabilities",
                num_top_classes=2,
            )

            result_summary = gr.Markdown(
                "Upload an image and click **Classify image** "
                "to see the prediction."
            )

    gr.Markdown("""
    ---

    **About this model**

    This application uses ResNet-18 transfer learning and
    recognizes two classes: ants and bees.

    For best results, upload a clear image similar to those
    used during training.
    """)

    gr.HTML("""
    <div id="footer">
        Built with PyTorch and Gradio · Ants vs Bees Classifier
    </div>
    """)

    # Run the prediction function when the button is clicked.
    classify_button.click(
        fn=classify_image,
        inputs=image_input,
        outputs=[prediction_output, result_summary],
    )


# Start the web server.
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        share=False,
    )
