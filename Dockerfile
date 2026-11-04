FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY stk ./stk
COPY LICENSE README.md ./
# STK_URL (your dedicated endpoint), STK_KEY (your dedicated API key), STK_FEATURE_KEY (GET /v1/keys/feature with the
# admin key: positions moved by your own key before they leave). STK_PII_NER=0 turns the English name model off
RUN useradd -r stk
USER stk
EXPOSE 8001
CMD ["uvicorn", "stk.edge:app", "--host", "0.0.0.0", "--port", "8001"]
