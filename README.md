<a href="https://huggingface.co/IzzulGod/GPT2-Small-Indonesian">
  <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-GPT2--Small--Indo-yellow" alt="Hugging Face">
</a>

[![Colab](https://img.shields.io/badge/Colab-T4%20GPU%20(Free)-orange?logo=googlecolab)](https://colab.research.google.com/)

# GPT2-Small Indonesian Chat Instruct-Tuned Model

An Indonesian conversational AI model fine-tuned from `GPT2-Small(124M Parameters)` using instruction-following techniques to enable chat-like interactions.

## 📋 Model Overview

This model transforms a base Indonesian GPT-2 text generator into a conversational chatbot capable of following instructions and engaging in question-answering dialogues in Bahasa Indonesia.

- **Base Model**: `GPT2-Small`
- **Fine-tuning Method**: SFT-LoRA (merged adapter)
- **Dataset**: `indonesian-nlp/wikipedia-id`, `FreedomIntelligence/evol-instruct-indonesian`, `FreedomIntelligence/sharegpt-indonesian`
- **Language**: Indonesian (Bahasa Indonesia)
- **Task**: Conversational AI / Chat Completion

## 🧪 Project Background

This model was fine-tuned as part of my personal learning journey in AI and LLMs. The training was done entirely on Google Colab (free tier, T4 GPU), as an exercise in building Indonesian conversational AI with limited resources.

## 🚀 Quick Start

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
model_path = "IzzulGod/GPT2-Small-Indonesian"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# Generate response
prompt = "User: Siapa presiden pertama Indonesia?\nAI:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Example Output

```
User: Siapa presiden pertama Indonesia?
AI: Presiden pertama Indonesia adalah Soekarno. Sukarno dikenal sebagai seorang pemimpin yang sangat dihormati dan dicintai oleh rakyatnya, terutama di kalangan rakyat Indonesia karena perananya dalam membentuk persatuan bangsa Indonesia. Dia juga dianggap sebagai sosok kunci bagi seluruh masyarakat Indonesia untuk mempertahankan kemerdekaan negara tersebut dari penjajahan Belanda.
```

## 🎯 Model Capabilities

- **Question Answering**: Responds to factual questions in Indonesian
- **Instruction Following**: Capable of following various instructions and tasks
- **Conversational Context**: Maintains context in chat-like interactions
- **Code Generation**: Can generate simple code snippets (R, Python, etc.) with Indonesian explanations

## 📊 Training Details

### Dataset

This model was trained on a dataset containing conversation data in the following format:

```json
[
  {
    "from": "human",
    "value": "Question or instruction in Indonesian"
  },
  {
    "from": "gpt", 
    "value": "Detailed response in Indonesian"
  }
]
```

### Training Configuration

The model was fine-tuned using LoRA (Low-Rank Adaptation) with aggressive parameter injection across key GPT-2 layers:

**LoRA Configuration:**
- `r`: 64 (rank)
- `lora_alpha`: 128
- `target_modules`: ["c_attn", "c_proj", "mlp.c_fc", "mlp.c_proj"]
- `lora_dropout`: 0.05
- `bias`: "none"

**Training Arguments:**
- `epochs`: 3
- `batch_size`: 16 per device
- `gradient_accumulation_steps`: 2
- `learning_rate`: 2e-4
- `scheduler`: cosine
- `weight_decay`: 0.01
- `fp16`: enabled

### Training Results

```
Final Training Loss: 2.692
Total Steps: 2,766
Training Time: ~1h 45m
```

The model showed consistent improvement with loss decreasing from 3.44 to 2.51 over the training period.

## 🔧 Advanced Usage

### Custom Generation Parameters

```python
# For more creative responses
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.3
)

# For more focused responses  
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.4,
    top_p=0.95,
    repetition_penalty=1.1
)
```

### Prompt Format

The model expects prompts in the following format:
```
User: [Your question or instruction in Indonesian]
AI:
```

## ⚠️ Limitations

- **Knowledge Base**: The base model was trained primarily on Wikipedia data: `indonesian-nlp/wikipedia-id` by [Cahya](https://huggingface.co/cahya), providing general factual knowledge but limited real-world conversational patterns
- **Training Data Scope**: Current fine-tuning focuses on general instruction-following and Q&A rather than natural daily conversations
- **Conversational Style**: Responses may feel formal or academic due to the Wikipedia-based foundation and instruction-tuned nature
- **Model Size**: Relatively small (124M Parameters), which may limit complex reasoning capabilities
- **Factual Accuracy**: Responses are generated based on training data and may not always be factually accurate or up-to-date
- **Language Optimization**: Best performance is achieved with Indonesian language inputs
- **Response Consistency**: May occasionally generate repetitive or inconsistent responses

## 🚀 Future Improvements

For enhanced conversational naturalness, consider:
- **Conversational Dataset Training**: Fine-tuning with Indonesian daily conversation datasets
- **Lighter LoRA Configuration**: Using more efficient LoRA parameters for conversation-specific training
- **Multi-turn Dialogue**: Training on multi-turn conversation data for better context handling
- **Informal Language Patterns**: Incorporating colloquial Indonesian expressions and casual speech patterns

## 📝 License

This model is released under the MIT License. See the LICENSE file for details.

## 📚 Citation

If you use this model in your research or applications, please cite:

```bibtex
@misc{izzulgod2025gpt2indochat,
  title     = {GPT2-Small Indonesian Chat Instruct-Tuned Model},
  author    = {IzzulGod},
  year      = {2025},
  howpublished = {\url{https://huggingface.co/IzzulGod/GPT2-Small-Indonesian}},
}
```
---

*Disclaimer: This model was developed as an experimental project for learning purposes. While it performs well on basic tasks, it may have limitations in reasoning and real-world usage.*
