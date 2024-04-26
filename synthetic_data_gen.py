import os
import time
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()  # This will load all the environment variables from the `.env` file.

# Initialize Groq client
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

def read_excel(file_path, sheet_name):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df

def extract_repo_name(url):
    # Extract the repository name from the URL
    repo_name = url.split("/")[-2] + '_' + url.split("/")[-1]
    return repo_name

def read_cobol_files(repo_path):
    # Read all COBOL files in the repository and return their paths and content
    files_content = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".cob") or file.endswith(".cbl") or file.endswith(".cpy"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    content = f.read()
                files_content.append((file_path, content))
    return files_content

def generate_description(repo_description, file_path, cobol_content):
    # Generate a detailed description for each file using Groq API
    messages = [
        {"role": "system", "content": "You are a detail-oriented assistant that generates descriptions for individual files within a repository."},
        {"role": "user", "content": f"Repository description: {repo_description}\n\nFile path: {file_path}\n\nPlease generate a detailed description of what this specific file does and how it fits into the overall repository:\n\n{cobol_content}"},
    ]
    retry_count = 0
    max_retries = 2  # Set maximum retries to 2
    while retry_count <= max_retries:
        try:
            response = groq_client.chat.completions.create(
                messages=messages,
                model="llama3-70b-8192",  # Adjust based on available model
                max_tokens=8192,
                temperature=0.25,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error occurred with llama3-70b-8192: {str(e)}")
            retry_count += 1
            if retry_count > max_retries:
                print("Maximum retries reached, moving to next file...")
                return "ERROR"
            print("Retrying with Mixtral-8x7b-32768 model...")
            try:
                response = groq_client.chat.completions.create(
                    messages=messages,
                    model="Mixtral-8x7b-32768",
                    max_tokens=32768,
                    temperature=0.25,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error occurred with Mixtral-8x7b-32768: {str(e)}")
                retry_count += 1
                if retry_count > max_retries:
                    print("Maximum retries reached, moving to next file...")
                    return "ERROR"
                print("Retrying in 120 seconds...")
                time.sleep(120)

# def generate_prompt(description):
    # Generate a user prompt for the LLM to generate the COBOL code
    messages = [
        {"role": "system", "content": "You are a user asking an AI to generate COBOL code based on a description."},
        {"role": "user", "content": f"Please generate a COBOL program based on the following description:\n\n{description}"},
    ]
    while True:
        try:
            response = groq_client.chat.completions.create(
                messages=messages,
                model="llama3-8b-8192",  # Adjust based on available model
                max_tokens=100,
                temperature=0.7,
            )
            break
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print("Retrying in 120 seconds...")
            time.sleep(120)
    prompt = response.choices[0].message.content.strip()
    return prompt

def update_excel(df, file_path):
    # Write the DataFrame back to an Excel file
    df.to_excel(file_path, index=False)

# Read the Excel file
df = read_excel(r"C:\Users\rochi\Desktop\CV\Code\Finetune COBOL\data\X-COBOL\X-COBOL\Information_Of_Repo.xlsx", sheet_name="Sheet3")
df = df[~df['url'].isna()]
# df = df.head(2)

# Read the existing updated_cobol_files.xlsx to check for already processed files
try:
    updated_df = read_excel("updated_cobol_files.xlsx", sheet_name="Sheet1")
    processed_files = set(updated_df['full_file_location'])
except FileNotFoundError:
    processed_files = set()

# Prepare a new DataFrame
new_df = pd.DataFrame(columns=['reponame', 'url', 'full_file_location', 'description', 'prompt', 'code'])

# Iterate over each row in the DataFrame
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing repositories"):
    repo_name = extract_repo_name(row['url'])
    repo_path = rf"C:\Users\rochi\Desktop\CV\Code\Finetune COBOL\data\X-COBOL\X-COBOL\COBOL_Files\{repo_name}"  # Replace with the actual path to COBOL files
    
    # Read COBOL files and process each file
    cobol_files = read_cobol_files(repo_path)
    for file_path, cobol_content in cobol_files:
        if file_path in processed_files:
            print(f"Skipping already processed file: {file_path}")
            continue
        
        print(f"Processing file: {file_path}")
        
        # Generate a detailed description for each file using the Groq API
        file_description = generate_description(row['description'], file_path, cobol_content)
        print(f"Generated description: {file_description}")
        
        # Generate a user prompt for the LLM to generate the COBOL code
        # prompt = generate_prompt(file_description)
        # print(f"Generated prompt: {prompt}")
        
        # Append to the new DataFrame
        new_df = pd.concat([new_df, pd.DataFrame([{
            'reponame': repo_name,
            'url': row['url'],
            'full_file_location': file_path,
            'description': file_description,
            # 'prompt': prompt,
            'code': cobol_content
        }])], ignore_index=True)
        
        # Update the Excel file intermittently
        update_excel(new_df, "updated_cobol_files.xlsx")

# Update the Excel file with the final data
update_excel(new_df, "updated_cobol_files.xlsx")