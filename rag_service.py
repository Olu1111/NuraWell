from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import logging
import sys
import uvicorn
from typing import Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instances
MODEL = None
TOKENIZER = None
CHROMA_DB = None

class ChatRequest(BaseModel):
    message: str
    max_new_tokens: int = 200
    temperature: float = 0.7
    timeout: Optional[int] = 30

def log_system_info():
    """Log critical system information"""
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")

def initialize_components():
    """Initialize model and vector DB with enhanced error handling"""
    global MODEL, TOKENIZER, CHROMA_DB
    
    try:
        log_system_info()
        
        # Model loading with fallback
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        logger.info("Loading Phi-3 model...")
        MODEL = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        ).to(device)
        logger.info("Model loaded successfully")
        
        TOKENIZER = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        
        logger.info("Loading vector database...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        CHROMA_DB = Chroma(
            persist_directory="./notebook/chroma_langchain_db",
            collection_name="pandora_conversations",
            embedding_function=embedding_model
        )
        logger.info("Vector DB loaded successfully")
        
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler with proper cleanup"""
    try:
        initialize_components()
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")

app = FastAPI(
    title="Phi-3 RAG Service",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {
        "status": "ready" if MODEL else "uninitialized",
        "device": str(MODEL.device) if MODEL else "N/A",
        "cuda_available": torch.cuda.is_available(),
        "chroma_ready": CHROMA_DB is not None
    }

@app.post("/chat")
async def generate_chat_response(request: ChatRequest):
    if not MODEL:
        raise HTTPException(status_code=503, detail="Service initializing")
    
    try:
        # Retrieve context
        results = CHROMA_DB.similarity_search(request.message, k=3)
        unique_contexts = list(set([doc.page_content for doc in results]))
        context = "\n".join([f"- {ctx}" for ctx in unique_contexts])
        
        # Prepare messages
        system_prompt = f"""You are a compassionate mental health assistant. 
        Consider these insights from similar situations:
        {context}
        
        Respond with:
        1. Empathetic validation
        2. Practical suggestions
        3. Professional recommendations when needed
        4. Hopeful encouragement"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message}
        ]
        
        # Tokenize inputs - critical fix here
        tokenized = TOKENIZER.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True
        ).to(MODEL.device)
        
        # Handle both dict and tensor outputs
        if isinstance(tokenized, dict):
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
        else:
            input_ids = tokenized
            attention_mask = tokenized.ne(TOKENIZER.pad_token_id).float()
        
        # Generate response
        outputs = MODEL.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=min(request.max_new_tokens, 400),
            do_sample=True,
            temperature=request.temperature,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=TOKENIZER.eos_token_id,
            eos_token_id=TOKENIZER.eos_token_id
        )
        
        # Decode response
        full_response = TOKENIZER.decode(
            outputs[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return {"response": full_response}
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Generation failed")   
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail="GPU memory exhausted")

if __name__ == "__main__":
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            timeout_keep_alive=300
        )
    except Exception as e:
        logger.critical(f"Service crashed: {str(e)}")
        sys.exit(1)