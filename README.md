<a href="https://huggingface.co/izzulgod/gpt2-indo-instruct-tuned">
  <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-GPT2--Small--Indo-yellow" alt="Hugging Face">
</a>

[![Colab](https://img.shields.io/badge/Colab-T4%20GPU%20(Free)-orange?logo=googlecolab)](https://colab.research.google.com/)


# GPT2-Small Indonesian Instruct-Tuned Model

An Indonesian conversational AI model fine-tuned from `GPT2-Small (124M Parameters)` using instruction-following techniques to enable natural chat-like interactions in Bahasa Indonesia.

## 📋 Model Overview

This model transforms a base Indonesian GPT-2 text generator into a conversational chatbot capable of following instructions and engaging in question-answering dialogues. The model has been specifically optimized for Indonesian language understanding and generation.

- **Base Model**: `GPT2-Small (124M Parameters)`
- **Fine-tuning Method**: SFT-LoRA (Supervised Fine-Tuning with Low-Rank Adaptation)
- **Training Datasets**: 
  - `indonesian-nlp/wikipedia-id` (knowledge base)
  - `FreedomIntelligence/evol-instruct-indonesian` (instruction following)
  - `FreedomIntelligence/sharegpt-indonesian` (conversational patterns)
- **Primary Language**: Indonesian (Bahasa Indonesia)
- **Task**: Conversational AI / Chat Completion
- **License**: MIT

## 🧪 Project Background

This model was developed as part of a personal learning journey in AI and Large Language Models (LLMs). The entire training process was conducted on **Google Colab's** free tier using **T4 GPU,** demonstrating how to build effective Indonesian conversational AI with limited computational resources.

The project focuses on creating an accessible Indonesian language model that can understand context, follow instructions, and provide helpful responses in natural Bahasa Indonesia.

## 🚀 Quick Start

### Installation

```bash
pip install transformers torch
```

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
model_path = "izzulgod/gpt2-indo-instruct-tuned"  
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# Create prompt
prompt = "User: Siapa presiden pertama Indonesia?\nAI:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>")
    )

# Decode response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 🎯 Model Capabilities

The model demonstrates strong performance across several Indonesian language tasks:

- **Question Answering**: Provides accurate responses to factual questions in Indonesian
- **Instruction Following**: Capable of understanding and executing various instructions and tasks
- **Conversational Context**: Maintains coherent context throughout chat-like interactions  
- **Code Generation**: Can generate simple code snippets (Python, R, etc.) with clear Indonesian explanations
- **Educational Content**: Explains complex concepts in accessible Indonesian language
- **Cultural Awareness**: Understands Indonesian cultural context and references

## 📊 Training Details

### Dataset Composition

The model was trained on a carefully curated combination of datasets to balance knowledge, instruction-following, and conversational abilities:

**Training Data Format:**
```json
[
  {
    "from": "human", 
    "value": "Question or instruction in Indonesian"
  },
  {
    "from": "gpt",
    "value": "Detailed and helpful response in Indonesian"
  }
]
```

### Training Configuration

The model was fine-tuned using LoRA (Low-Rank Adaptation) technique, which allows efficient training while preserving the base model's capabilities:

**LoRA Configuration:**
- **Rank (r)**: 64 - Higher rank for better adaptation capacity
- **Alpha**: 128 - Scaling factor for LoRA weights  
- **Target Modules**: `["c_attn", "c_proj", "mlp.c_fc", "mlp.c_proj"]` - Key transformer components
- **Dropout**: 0.05 - Regularization to prevent overfitting
- **Bias**: "none" - Focus adaptation on weight matrices

**Training Hyperparameters:**
- **Epochs**: 3 - Sufficient for convergence without overfitting
- **Batch Size**: 16 per device - Optimized for T4 GPU memory
- **Gradient Accumulation**: 2 steps - Effective batch size of 32
- **Learning Rate**: 2e-4 - Conservative rate for stable training
- **Scheduler**: Cosine annealing - Smooth learning rate decay
- **Weight Decay**: 0.01 - L2 regularization
- **Mixed Precision**: FP16 enabled - Memory and speed optimization

### Training Progress

The model showed consistent improvement throughout training:

```
Training Progress (5535 total steps over 3 epochs):
Step	Training Loss
200	    3.533500  # Initial high loss
400	    2.964200  # Rapid initial improvement
...
4000	2.416200  # Stable convergence
...
5400	2.397500  # Final optimized loss

Final Metrics:
- Training Loss: 2.573
- Training Time: 3.5 hours
- Samples per Second: 14.049
- Total Training Samples: ~177k
```

The steady decrease from 3.53 to 2.39 demonstrates effective learning and adaptation to the Indonesian instruction-following task.

## 🔧 Advanced Usage

### Generation Parameter Tuning

**For Creative Responses:**
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=256,      # Longer responses
    temperature=0.8,         # More randomness
    top_p=0.9,              # Diverse vocabulary
    repetition_penalty=1.2   # Avoid repetition
)
```

**For Focused/Factual Responses:**
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=128,      # Concise responses  
    temperature=0.6,         # More deterministic
    top_p=0.95,             # High-quality tokens
    repetition_penalty=1.1   # Mild repetition control
)
```

### Prompt Engineering

**Recommended Format:**
```
User: [Your question or instruction in Indonesian]
AI: [Expected response starts here]
```

## ⚠️ Limitations and Considerations

**Knowledge Limitations:**
- **Training Data Cutoff**: Knowledge is limited to the training datasets, primarily Wikipedia-based information
- **Factual Accuracy**: Generated responses may not always be factually accurate or up-to-date
- **Real-time Information**: Cannot access current events or real-time data

**Technical Limitations:**
- **Model Size**: With 124M parameters, complex reasoning capabilities are limited compared to larger models
- **Context Length**: Limited context window may affect very long conversations
- **Language Specialization**: Optimized primarily for Indonesian; other languages may produce suboptimal results

**Response Characteristics:**
- **Formality**: Responses may occasionally sound formal due to Wikipedia-based training data
- **Consistency**: May generate repetitive patterns or inconsistent information across sessions
- **Cultural Nuances**: While trained on Indonesian data, may miss subtle cultural references or regional variations

## 🚀 Future Development Roadmap

**Short-term Improvements:**
- Fine-tuning with more diverse conversational datasets
- Integration of current Indonesian news and cultural content
- Specialized domain adaptations (education, healthcare, business)

## 📝 License

This model is released under the MIT License, Please see the LICENSE file for complete terms.

## 🙏 Acknowledgments

- **Base Model**: Thanks to [Cahya](https://huggingface.co/cahya) for the Indonesian GPT-2 base model
- **Datasets**: FreedomIntelligence team for Indonesian instruction and conversation datasets
- **Infrastructure**: Google Colab for providing accessible GPU resources for training

---

**Disclaimer**: This model was developed as an experimental project for educational and research purposes. While it demonstrates good performance on various tasks, users should validate outputs for critical applications and be aware of the limitations outlined above.
