import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def generate_response(query, max_new_tokens=400, temperature=0.7):
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

    # Load the Chroma database
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    chroma_db = Chroma(
        collection_name="pandora_conversations",
        embedding_function=embedding_model,
        persist_directory="./notebook/chroma_langchain_db"
    )

    # Retrieve and deduplicate context
    results = chroma_db.similarity_search(query, k=3)
    unique_contexts = list(set([doc.page_content for doc in results]))
    context = "\n".join([f"- {ctx}" for ctx in unique_contexts])
    
    # Improved prompt template
    system_prompt = f"""You are a compassionate mental health assistant. 
    Consider these insights from similar situations (adapt appropriately):
    {context}
    
    Key principles:
    1. Be wise, empathetic and understanding
    2. Suggest practical steps
    3. Recommend professional help when appropriate
    4. Maintain loving, caring, but direct tone"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    
    # Tokenize with proper handling
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_attention_mask=False
    ).to(model.device)
    
    # Create optimized attention mask
    attention_mask = inputs.ne(tokenizer.pad_token_id).float()
    
    # Enhanced generation config
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Clean output processing
    full_response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    # Post-processing for clinical safety
    if "suicide" in query.lower() or "self-harm" in query.lower():
        disclaimer = "\n\n[Important] If you're having thoughts of harming yourself, please call the National Suicide Prevention Lifeline at 988 (US) or your local crisis hotline immediately."
        return full_response + disclaimer
    
    return full_response

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
        response = generate_response(query)
        print(response)
