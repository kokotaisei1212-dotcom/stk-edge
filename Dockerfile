FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY stk ./stk
COPY LICENSE README.md ./
# STK_URL (default https://api.stk-evidence.com), STK_KEY (your STK key), STK_PII_NER=0 turns the English name model off
RUN useradd -r stk
USER stk
EXPOSE 8001
CMD ["uvicorn", "stk.edge:app", "--host", "0.0.0.0", "--port", "8001"]
