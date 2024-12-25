The repository contains a dataset we constructed for a multi-label syndrome differentiation task in Traditional Chinese Medicine (TCM).

**Due to the sensitivity of medical data, we will make the dataset publicly available as soon as we obtain permission from the hospital.**

For now, we have only provided three sample entries along with the code for classification using large model inference capabilities, for those interested in exploring the task.

#### **File Explanation**

Specifically, we used three open-source large models: **Huatuo, Bencao, qwen** and **LLaMA 3**, as well as a closed-source high-performance model like **ChatGPT** for inference. We placed the corresponding code in Python files named after each model.

**Model Link**

1. https://github.com/Borororo/ZY-BERT
2. https://github.com/CrazyBoyM/llama3-Chinese-chat
3. https://github.com/FreedomIntelligence/HuatuoGPT
4. https://github.com/QwenLM/Qwen1.5
5. https://github.com/baichuan-inc/Baichuan2
6. https://github.com/yao8839836/tcm_bert
7. https://github.com/ymcui/Chinese-BERT-wwm
8. https://modelscope.cn/models/ZhipuAI/ChatGLM-6B/summary

#### Environment

All codes are tested under Python 3.8, PyTorch 1.12.1.

numpy == 1.21.5

pillow == 9.4.0

scikit-learn ==1.0.2

transformers == 4.24.0

tqdm == 4.65.2

opt_einsum == 3.3.0

**RUN CODE**

You only need to run the Python script to perform inference. For example, if you want to use ChatGPT for inference, simply place your API key in the appropriate location in the `chat_openai4.py` file, then

```
python  chat_openai4.py
```

to generate the corresponding predictions from the model.

For the framework of our model, just run `main.py` directly

For a simple deep learning model, first put the corresponding data into `word2vec.py`  to obtain the corresponding word vector, then put the obtained word vector text into the current folder, and finally run the `Bi-GRU.py` corresponding to the Bi-GRU model to get the result.

**Sample Data and ALL Knowledge**

Additionally, we have provided three sample data points in `sample_data.txt`. To further assist future work, we have also placed explanation files for each diagnosis type in the `knowledge.json` file. Each entry in the file details the manifestations of a specific diagnosis type and the required medications.

**Stay tuned!**