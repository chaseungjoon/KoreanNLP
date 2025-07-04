# RAG_7 Model Pipeline

### Core Components

1.  **Language Model**: Utilizes `beomi/Llama-3-Open-Ko-8B-Instruct-preview`, a large language model optimized for Korean. It's loaded with 4-bit quantization to reduce memory usage.
2.  **Retriever**: A sophisticated `HybridRerankRetriever` that combines sparse and dense retrieval methods to find the most relevant documents for a given question.
    *   **Sparse Retrieval**: `BM25Okapi` is used for keyword-based document matching.
    *   **Dense Retrieval**: `snunlp/KR-SBERT-V40K-klueNLI-augSTS` (SBERT) is used to find semantically similar documents by comparing vector embeddings.
    *   **Reranking**: A `bongsoo/klue-cross-encoder-v1` model is used to rerank the documents retrieved by the hybrid method, further improving relevance.
3.  **Prompt Engineering**: A structured prompt is used, containing a system message, the retrieved reference documents, the question type, the question itself, and specific instructions based on the question type

### Training Pipeline

1.  **Initialization**: The script loads the reference documents, which form the knowledge base. The `HybridRerankRetriever` is initialized by encoding all reference passages into vector embeddings for SBERT and tokenizing them for BM25.
2.  **Model Preparation**: The base language model is loaded and prepared for training using PEFT (Parameter-Efficient Fine-Tuning) with LoRA (Low-Rank Adaptation). This allows for efficient fine-tuning by only updating a small number of parameters.
3.  **Data Loading**: Training and validation datasets are loaded from JSON files.
4.  **Dataset Creation**: A `KoreanRAGDataset` is created. For each training sample, it generates a prompt, retrieves relevant context using a simplified SBERT search, and combines the prompt with the ground-truth answer to create the final input for the model.
5.  **Fine-Tuning**: The model is fine-tuned using the `transformers.Trainer`. The goal is to teach the model how to generate answers based on the provided context and question format.
6.  **Saving Artifacts**: After training, the script saves the fine-tuned LoRA model adapter, the tokenizer, and the retriever's pre-computed data (`retriever.pt`) to an output directory.

### Inference Pipeline

1.  **Model Loading**: The base language model is loaded, and the fine-tuned LoRA adapter is applied to it.
2.  **Retriever Loading**: The pre-computed retriever data (embeddings and tokenized passages) is loaded from the `retriever.pt` file.
3.  **Inference Loop**: The script iterates through the test dataset. For each question:
    a.  **Hybrid Retrieval & Reranking**: The `HybridRerankRetriever` is used to fetch the top 5 most relevant documents from the knowledge base. It first gets a larger set of candidate documents using a weighted score from BM25 and SBERT, and then the cross-encoder reranks these candidates to select the final set.
    b.  **Prompt Generation**: A detailed prompt is constructed using the retrieved documents and the question.
    c.  **Answer Generation**: The model generates an answer based on the prompt. Generation parameters like `temperature` and `repetition_penalty` are used to control the output quality.
4.  **Post-processing**: The generated text is cleaned and formatted to match the required output style.
5.  **Output**: The final predictions are written to a `submission.jsonl` file.
