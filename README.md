# RyanairRoastBot
Fine-tuning an LLM to Generate Controlled Sarcastic Responses

## Overview

RyanairRoastBot is an experimental natural language generation project that explores how the behavior of a large language model can be intentionally shifted away from its default polite and obedient style toward a sarcastic, snarky, brand-inspired tone, inspired by Ryanair’s well-known social media communication.

The main objective of this project is to demonstrate that large language models can be fine-tuned not only for correctness and helpfulness, but also for controlled personality and tone, while still enforcing strict safety constraints.

Although the model produces sarcastic replies, it is explicitly designed to remain safe:

- No hate speech  
- No slurs or harassment  
- No threats  
- No profanity  
- Serious and helpful responses for emergency, medical, or accessibility-related situations  

This project was developed collaboratively .

---

## Project Goals

- Fine-tune an instruction-following LLM using LoRA adapters  
- Explore non-standard assistant behavior (sarcastic instead of obedient)  
- Generate short, airline-style roast replies  
- Enforce safety constraints despite humorous tone  
- Provide an interactive inference interface  

---

## Model and Tools

- **Base model:** `unsloth/llama-3-8b-Instruct-bnb-4bit`
- **Fine-tuning method:** LoRA (Parameter-Efficient Fine-Tuning)
- **Quantization:** 4-bit
- **Libraries and frameworks:**
  - Unsloth
  - Hugging Face Transformers
  - TRL (SFTTrainer)
  - PEFT
  - PyTorch
- **Interface:** Gradio

---


The training dataset is private and therefore not included in this repository.

---

## Dataset Format (Private)

The model was trained using a private dataset in JSON Lines (`.jsonl`) format.

Each training example follows this structure:

```json
{
  "instruction": "Write a Ryanair-style roast reply. Length: 1–2 sentences.",
  "input": "I want a refund",
  "output": "We’ll process that refund right after we discover a business model where refunds make sense."
}
````

### Fields

* `instruction`: generation instruction
* `input`: user complaint
* `output`: expected sarcastic reply

---

## Prompting Strategy

Each training example is converted into a structured chat format:

* **System prompt** defines tone, personality, and safety rules
* **User message** contains the instruction and the complaint
* **Assistant message** contains the expected response

This structure ensures alignment between training and inference.

---

## Safety Design

Despite the sarcastic tone, the following rules are enforced:

* No slurs or hate speech
* No harassment or threats
* No profanity
* Maximum length of 1–2 sentences

If the user message involves:

* safety-related situations
* medical concerns
* disability or accessibility needs
* emergencies

the model must respond seriously and helpfully, without sarcasm.

---


## Inference and Demo


```bash
python web_app.py
```

A Gradio web interface will launch locally and allow interactive testing of user complaints.

---

## Authors

This project was developed collaboratively on a remote machine by:

* **Marwa Founi**
* **Zayneb Fathalli**

Both authors contributed equally to:

* dataset design
* prompt engineering
* fine-tuning strategy
* evaluation and testing
* inference interface development

---

## Disclaimer

This project is intended for educational and research purposes only.

It is not affiliated with Ryanair and does not represent the company or its official communication.

```


