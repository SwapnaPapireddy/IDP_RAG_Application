# Intelligent Document Processing & RAG — Without Amazon Bedrock

## Architecture
1. Streamlit local web UI accepts a PDF or image.
2. The original file is uploaded to an **existing** Amazon S3 bucket.
3. Amazon Textract performs OCR (asynchronous text detection for PDFs; synchronous for images).
4. Text is cleaned and split into overlapping chunks with document and page metadata.
5. `sentence-transformers/all-MiniLM-L6-v2` creates embeddings on your machine.
6. FAISS stores and searches vectors locally.
7. A Hugging Face Transformers model (`google/flan-t5-base` by default) runs locally and answers from retrieved context.
8. Streamlit displays the answer and source chunks/pages.

### Architecture image

"<img width="708" height="638" alt="image" src="https://github.com/user-attachments/assets/7665c609-3cd5-417e-81b3-8c81d6db1673" />


</td>

<td align="center">

The vector index is local FAISS.

## Setup (Windows PowerShell)
Use Python 3.10–3.12 for best compatibility with PyTorch/FAISS.
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```
Set `AWS_REGION` to your bucket's region and `S3_BUCKET` to the name of an existing bucket. Do not put access keys in `.env`. Use your configured AWS CLI/SSO credentials.

If using the `intellegent_rag` SSO profile:
```powershell
aws sso login --profile intellegent_rag
$env:AWS_PROFILE="intellegent_rag"
aws sts get-caller-identity
streamlit run app.py
```
### Live Demo

https://screenrec.com/share/YT2zkm89P6

### output image

<img width="1300" height="593" alt="image" src="https://github.com/user-attachments/assets/d415826d-29c0-4f77-8b52-827f4bdcaf02" />


## IAM permissions
The AWS identity needs S3 `s3:PutObject` on `arn:aws:s3:::YOUR_BUCKET/uploads/*` and Textract `textract:StartDocumentTextDetection`, `textract:GetDocumentTextDetection`, and `textract:DetectDocumentText`. Grant only the minimum permissions needed. This code does not create buckets or IAM roles.

## Cost and account safety
- Bedrock is not used.
- S3 and Textract may still incur charges; check your plan, credits, region, and Textract service access before uploading.
- Hugging Face models download to your computer and run locally; no paid hosted inference endpoint is created.
- If your Free plan blocks Textract or shows a subscription-required error, stop rather than upgrading just to test.
- Clear local index does not delete S3 objects.

## Files
- `app.py`: Streamlit UI
- `pipeline.py`: S3 upload, Textract OCR, chunking, local embeddings, FAISS retrieval, local Hugging Face generation
- `.env.example`: configuration template
- `requirements.txt`: dependencies

## Limitations
This is a small single-machine prototype. FAISS index and metadata live in `data/` locally. `flan-t5-base` is relatively small and may be less capable than larger hosted models. Use only documents you are authorized to process.
