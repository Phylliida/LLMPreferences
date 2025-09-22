import setuptools

setuptools.setup(
    name = "llmprefs",
    version = "0.0.1",
    author = "Phylliida",
    author_email = "phylliidadev@gmail.com",
    description = "Studying preferences of LLMs",
    url = "https://github.com/Phylliida/LLMPreferences.git",
    project_urls = {
        "Bug Tracker": "https://github.com/Phylliida/LLMPreferences/issues",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages = setuptools.find_packages(),
    python_requires = ">=3.6",
    install_requires = ["vllm[tensorizer]", "langchain", "flashinfer-python", "pandas", "seaborn", "numpy", "torch", "ujson", "setuptools", "pyarrow", "markdownify", "pytz", "huggingface-hub", "langchain", "transformers", "pingouin", "scipy", "safetytooling @ git+https://github.com/safety-research/safety-tooling.git@abhay/tools"]
)
