# Stable Diffusion for ads

This project is a personal experiment in fine-tuning **Stable Diffusion v1.5** to generate stylish, theme-specific advertisement banners using **DreamBooth** with **LoRA**. The goal was to make the model better at producing visually compelling ad posters with clean compositions.

## Running it

### 1. Clone the Repository
```bash
git clone https://github.com/dhrm1k/stable-diff-for-ads.git
cd stable-diff-for-ads
```

### 2. Set Up the Environment
Create a virtual environment (optional but recommended) and install dependencies:
```python
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Code in the jupyter file runs as it is on google cloud. For the file modal-stable-diff-ad.py you need to configure [modal](http://modal.com/) first and then running it will work as expected.

## 📓 Notes
This repo is a work in progress and will evolve as more experiments are conducted. Read more about it [here](https://dhrm1k.github.io/experimenting-with-stable-diffusion.html). Also, I have few more details in ```INFO.md```.
Feel free to fork or suggest improvements in the issues.