# 🤖 Fine-Tuning GPT-2 com Dataset Amazon-1.3M

Projeto de Fine-Tuning do modelo GPT-2 para responder perguntas sobre produtos da Amazon, utilizando o dataset LF-Amazon-1.3M.

## 📋 Descrição

Este projeto treina o modelo GPT-2 da Hugging Face para gerar respostas contextualizadas sobre produtos Amazon. O modelo aprende a responder perguntas no formato: "O que é [produto]?" com base em títulos e descrições de produtos reais.

## 🎯 Objetivo

Criar um modelo de linguagem especializado em produtos Amazon, capaz de:

- Responder perguntas sobre características de produtos
- Gerar descrições contextualizadas
- Demonstrar conhecimento específico do domínio e-commerce
- 
## 🚀 Tecnologias Utilizadas

- Python 3.x  
- Pandas  
- NumPy  
- Matplotlib / Seaborn  
- Scikit-learn  
- Jupyter Notebook  

## 📂 Estrutura do Projeto

```
Tech-Challenge.ipynb   # Notebook principal com todo o código e análises
README.md              # Documentação do projeto
data/                  # (Opcional) Diretório para os datasets usados
models/                # (Opcional) Diretório para modelos treinados
```

## 🧠 Principais Etapas Implementadas

# Fine-Tuning GPT-2 com Dataset Amazon-1.3M

**Objetivo:** Treinar GPT-2 para responder perguntas sobre produtos Amazon.

**GPU:** Otimizado para NVIDIA RTX 3070 (8GB VRAM)


```python
# 1️⃣ Verificar GPU
import torch

assert torch.cuda.is_available(), "❌ GPU não detectada! Este notebook requer GPU."

print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"🔧 CUDA: {torch.version.cuda}")
```

    ✅ GPU: NVIDIA GeForce RTX 3070 Laptop GPU
    💾 VRAM: 8.0 GB
    🔧 CUDA: 11.8



```python
# 2️⃣ Imports
import json
import math
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
print("📦 Bibliotecas carregadas")
```

    C:\Users\niloa\AppData\Roaming\Python\Python311\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    W1007 11...
*(trecho do notebook truncado para resumo)*

## 📊 Resultados

Os resultados e métricas de desempenho estão documentados ao longo do notebook, incluindo gráficos e tabelas comparativas.

## 🧩 Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/tech-challenge.git
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o notebook:
   ```bash
   jupyter notebook Tech-Challenge.ipynb
   ```

## ✍️ Mais Sobre o Projeto

- Esse projeto faz parte dos de estudos da PosTech FIAP
- Curso: IA Para DEVS
- Tech Challenge Fase 2

## ✍️ Integrantes do Grupo

- João Pedro Noleto – RA: 363789
- Danilo Almeida de Sousa – RA: 362639
- Allan dos Santos – RA: 362388
- Thiago Silva França – RA: 361249
