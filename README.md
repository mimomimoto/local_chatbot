# local chatbot with pdf
## Setup environment
```bash
git clone https://github.com/mimomimoto/elyza_cpu.git

conda create -n local_chatbot python==3.10.15
activate local_chatbot
pip install -r requirement.txt
```

## Install model
```bash
wget -P models/Llama-3-ELYZA-JP-8B-GGUF https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B-GGUF/resolve/main/Llama-3-ELYZA-JP-8B-q4_k_m.gguf
```
create Modelfile

```txt
FROM ./Llama-3-ELYZA-JP-8B-q4_k_m.gguf
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|reserved_special_token"
```

create Ollama model
```bash
ollama create elyza:jp8b-gguf -f Modelfile
```

## Create database
modify OpenAI API key in 'create_database.py'
```python
API_KEY = "your api key"
```

```bash
python create_database.py 'pdf_path'
```

## Run local chatbot
```bash
streamlit run main.py elyza:jp8b-gguf 'pdf_path'
```