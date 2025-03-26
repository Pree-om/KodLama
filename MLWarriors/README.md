# AI-Powered Waste Classification and Energy Conversion

## Overview
This project aims to classify waste items using a Convolutional Neural Network (CNN) and generate an energy conversion method using a Natural Language Processing (NLP) model. The system integrates image classification with text summarization to provide efficient waste-to-energy solutions.

## Features
- **CNN-based Waste Classification**: Uses a ResNet50 model to classify waste items.
- **NLP-based Energy Conversion Summarization**: Leverages Facebook's BART model to generate concise energy conversion methods.
- **Gradio Web Interface**: Deploys the system for easy interaction and testing.

## Dataset
The model is trained using the **Recyclable and Household Waste Classification Dataset** from Kaggle ([alistairking/recyclable-and-household-waste-classification](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification)).

## Installation
### Prerequisites
- Python 3.8+
- Google Colab (Recommended)
- PyTorch
- Transformers (Hugging Face)
- torchvision
- PIL (Pillow)
- Matplotlib
- Gradio

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/ai-waste-classification.git
   cd ai-waste-classification
   ```
2. Install dependencies:
   ```sh
   pip install torch torchvision transformers gradio matplotlib
   ```
3. Download the dataset using KaggleHub:
   ```python
   import kagglehub
   path = kagglehub.dataset_download("alistairking/recyclable-and-household-waste-classification")
   ```

## Training the Model
1. **Dataset Preparation**: The dataset is split into training (60%), validation (20%), and testing (20%).
2. **CNN Model Training**: A ResNet50-based model is trained for waste classification.
3. **NLP Model Integration**: The BART model provides waste-to-energy conversion summaries.

Run the training script:
```sh
python trainingtwo.py
```

## Using the Gradio Interface
Launch the Gradio web app for classification and energy conversion:
```python
import gradio as gr

def gradio_pipeline(image):
    classified_waste = classify_waste(image, image_model)
    energy_conversion = waste_to_energy(classified_waste, summarizer)
    return classified_waste, energy_conversion

interface = gr.Interface(
    fn=gradio_pipeline,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Text(label="Classified Waste"), gr.Text(label="Energy Conversion Method")],
    title="AI Waste Classification & Energy Conversion"
)
interface.launch(share=True)
```

## Results & Performance
- **Training Accuracy**: ~90%
- **Validation Accuracy**: ~85%
- **Test Accuracy**: ~83%
- **Waste-to-Energy Explanations**: Summarized using BART NLP

## Future Work
- Expand dataset for better generalization.
- Improve classification accuracy with advanced architectures.
- Integrate real-time mobile app support.

## License
This project is licensed under the MIT License.

## Contact
For any inquiries, feel free to reach out via email: `your-email@example.com`.
