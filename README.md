COBOL Fine-tuning and Code Generation
=====================================

This repository contains code for fine-tuning a language model on COBOL code and generating detailed descriptions and prompts for COBOL files using the Groq API.

Prerequisites
-------------

Before running the code, make sure you have the following:

-   Python 3.x installed
-   Required Python packages installed (see `requirements.txt`)
-   Groq API key
-   Weights and Biases API key (optional, for logging)

Setup
-----

1.  Clone the repository:

bash

Copy code

`git clone https://github.com/rochismandatta/Fine_tune_LLM.git
cd Fine_tune_LLM`

1.  Install the required Python packages:

bash

Copy code

`pip install -r requirements.txt`

1.  Create a `.env` file in the project root and add your API keys:

Copy code

`GROQ_API_KEY=your-groq-api-key
WANDB_API_KEY=your-wandb-api-key`

Usage
-----

### Fine-tuning the Model

To fine-tune the language model on COBOL code:

1.  Prepare your training and evaluation datasets in JSONL format and place them in the `data/SQL` directory.
2.  Run the `finetune.py` script:

bash

Copy code

`python finetune.py`

The script will download the pre-trained model if it's not already available locally, fine-tune the model on the provided dataset, and save the fine-tuned model in the `models/phi-3-mini-4k-instruct-gguf-finetuned` directory.

### Generating Descriptions and Prompts

To generate detailed descriptions and prompts for COBOL files:

1.  Update the `Information_Of_Repo.xlsx` file with the repository URLs and descriptions.
2.  Place the COBOL files in the `data/X-COBOL/X-COBOL/COBOL_Files` directory, with each repository having its own subfolder.
3.  Run the `generate_descriptions.py` script:

bash

Copy code

`python generate_descriptions.py`

The script will process each COBOL file, generate detailed descriptions using the Groq API, and save the results in the `updated_cobol_files.xlsx` file.

File Structure
--------------

-   `finetune.py`: Script for fine-tuning the language model on COBOL code.
-   `generate_descriptions.py`: Script for generating detailed descriptions and prompts for COBOL files.
-   `data/SQL`: Directory for storing the training and evaluation datasets for fine-tuning.
-   `data/X-COBOL/X-COBOL/COBOL_Files`: Directory for storing the COBOL files.
-   `models`: Directory for storing the pre-trained and fine-tuned models.
-   `Information_Of_Repo.xlsx`: Excel file containing repository URLs and descriptions.
-   `updated_cobol_files.xlsx`: Excel file containing the generated descriptions and prompts for COBOL files.

Contributing
------------

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

License
-------

This project is licensed under the [MIT License](LICENSE).

Acknowledgements
----------------

-   [Groq API](https://www.groq.com/) for providing the language model capabilities.
-   [Hugging Face](https://huggingface.co/) for the pre-trained models and libraries.
-   [Weights and Biases](https://wandb.ai/) for experiment tracking and logging.
