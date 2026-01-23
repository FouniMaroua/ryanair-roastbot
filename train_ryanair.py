import unsloth
from unsloth import FastLanguageModel

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer


MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"

SYSTEM_PROMPT = (
    "You are RyanairRoastBot. Tone: snarky, dry, sarcastic. "
    "Rules: 1–2 sentences. Must address the user's complaint directly. "
    "Must mention required keywords when provided. "
    "No slurs, no hate, no threats, no harassment. Avoid profanity. "
    "If the situation is safety-critical, medical, disability/accessibility, or emergency: "
    "be serious and helpful, no roast."
)


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = load_dataset("json", data_files="train.jsonl", split="train")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    def formatting_prompts_func(examples):
        texts = []
        for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{inst}\n\nUser complaint: {inp}"},
                {"role": "assistant", "content": out},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    train_ds = dataset["train"].map(
        formatting_prompts_func,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    eval_ds = dataset["test"].map(
        formatting_prompts_func,
        batched=True,
        remove_columns=dataset["test"].column_names,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=1024,
        packing=True,
        args=TrainingArguments(
            output_dir="ryanair_model",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=3,
            learning_rate=1e-4,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",

            logging_steps=10,

            eval_strategy="steps",
            eval_steps=50,

            save_strategy="steps",
            save_steps=50,

            report_to="none",

            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
        ),
    )

    print(" Training started ")
    trainer.train()

    print(" Saving model")
    model.save_pretrained("ryanair_final_model")
    tokenizer.save_pretrained("ryanair_final_model")
    print(" Saved to ryanair_final_model")


if __name__ == "__main__":
    main()
