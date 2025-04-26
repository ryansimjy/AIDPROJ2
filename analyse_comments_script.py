
import pandas as pd
import re
import ast
from deep_translator import GoogleTranslator
from transformers import pipeline, logging as hf_logging
import matplotlib.pyplot as plt

import nltk

import os # For changing directory
import torch # To check GPU availability
import time # Import time for potential delay

print("Libraries imported.")

# Check if GPU is available
if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    target_device = 0
else:
    print("WARNING: No GPU detected by PyTorch. Pipelines will run on CPU (which will be very slow).")
    target_device = -1

# Optional: Suppress verbose logging from transformers if desired
# hf_logging.set_verbosity_error()

# Optional: Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' tokenizer found.")
except LookupError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    try:
        nltk.download('punkt', quiet=True)
        print("NLTK 'punkt' downloaded successfully.")
        nltk.data.find('tokenizers/punkt')
        print("NLTK 'punkt' verified after download.")
    except Exception as download_e:
        print(f"ERROR: Failed to download or verify NLTK 'punkt' data after attempting: {download_e}")
        print("N-gram analysis might fail if tokenization relies on 'punkt'.")
# =========================================
# 3. Configuration
# =========================================
# Define file names, categories, and parameters.
CSV_FILE_PATH = "apple_vision_pro_comments.csv"
COMMENT_COLUMN_NAME = "comments"
MAX_COMMENTS_TO_PROCESS = 50 # Limit processing to the first 50 comments

# Define categories for Zero-Shot Classification and Bar Chart
CATEGORIES = ["Price", "Material", "Comfort", "Headstrap", "Battery life", "Connectivity", "Weight"]

# Output files (will be saved in Drive) - Updated names for the limited set
OUTPUT_CSV_FILE = f"classified_first_{MAX_COMMENTS_TO_PROCESS}_comments_by_category.csv"
OUTPUT_NEGATIVE_CHART_FILE = f"negative_comments_per_category_first_{MAX_COMMENTS_TO_PROCESS}.png"
OUTPUT_POSITIVE_CHART_FILE = f"positive_comments_per_category_first_{MAX_COMMENTS_TO_PROCESS}.png" # Added filename for positive chart

print("Configuration set:")
print(f" - Input CSV (in Drive): {CSV_FILE_PATH}")
print(f" - Comment Column: {COMMENT_COLUMN_NAME}")
print(f" - Processing AT MOST the first {MAX_COMMENTS_TO_PROCESS} comments.") # Updated message
print(f" - Categories: {', '.join(CATEGORIES)}")
print(f" - Output CSV (in Drive): {OUTPUT_CSV_FILE}")
print(f" - Output Negative Chart (in Drive): {OUTPUT_NEGATIVE_CHART_FILE}")
print(f" - Output Positive Chart (in Drive): {OUTPUT_POSITIVE_CHART_FILE}")

# =========================================
# 4. Define Helper Functions
# =========================================

def safe_literal_eval(val):
    """Safely evaluate a string literal (like a list) or return empty list on error."""
    try:
        if isinstance(val, str):
            if val.strip().startswith('[') and val.strip().endswith(']'):
                return ast.literal_eval(val)
            else:
                 return [val] # Treat plain string as a single-item list
        elif isinstance(val, list):
            return val
        else:
            return []
    except (ValueError, SyntaxError, TypeError):
        return []

def clean_text(text):
    """Cleans the input text."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'\<.*?\>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = text.replace('\n', ' ').replace('’', "'")
    text = text.replace('€', ' euros ').replace('£', ' gbp ').replace('$', ' dollar ').replace('%', ' percent ')
    text = text.replace("'", " ").replace('&quot;', ' ')
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Helper functions defined.")

# =========================================
# 5. Load and Prepare Data
# =========================================

# --- Load comments from CSV ---
print(f"Loading comments from '{CSV_FILE_PATH}' in Google Drive...")
df_comments = None
try:
    if not os.path.exists(CSV_FILE_PATH):
         raise FileNotFoundError(f"The file '{CSV_FILE_PATH}' was not found in the current directory: {os.getcwd()}")
    df_comments = pd.read_csv(CSV_FILE_PATH)
    if COMMENT_COLUMN_NAME not in df_comments.columns:
         raise ValueError(f"Column '{COMMENT_COLUMN_NAME}' not found in the CSV file.")
    print(f"Successfully loaded {len(df_comments)} rows.")
except FileNotFoundError as e:
    print(f"ERROR: {e}")
except ValueError as e:
    print(f"ERROR: {e}")
except Exception as e:
    print(f"ERROR: Failed to load CSV file. Details: {e}")

# --- Extract and flatten comments, then take the first MAX_COMMENTS_TO_PROCESS ---
comment_list = []
if df_comments is not None:
    print("Extracting and flattening comments...")
    all_comments_extracted = []
    for item in df_comments[COMMENT_COLUMN_NAME].dropna():
        evaluated_item = safe_literal_eval(item)
        if isinstance(evaluated_item, list):
            all_comments_extracted.extend([str(comment).strip() for comment in evaluated_item if str(comment).strip()])

    # --- Limit to the first MAX_COMMENTS_TO_PROCESS comments ---
    comment_list = all_comments_extracted[:MAX_COMMENTS_TO_PROCESS]
    print(f"Extracted {len(all_comments_extracted)} total comments, processing the first {len(comment_list)} (max {MAX_COMMENTS_TO_PROCESS}).")
    if not comment_list:
        print("No comments found or extracted within the limit.")

# --- Translate to English ---
if comment_list:
    print(f"Translating {len(comment_list)} comments (if needed)...")
    translated_comments = []
    count = 0
    total = len(comment_list)
    start_time = time.time()
    for i, comment in enumerate(comment_list):
        try:
            # Uncomment the line below if you hit translation rate limits/errors
            # time.sleep(0.1)
            translated = GoogleTranslator(source="auto", target="en").translate(comment)
            translated_comments.append(translated if translated else comment)
        except Exception as e:
            # print(f"Warning: Translation failed for comment {i+1}. Keeping original. Error: {e}")
            translated_comments.append(comment) # Keep original on error
        count += 1
        # Print progress periodically (more frequent for smaller batches)
        if count % 25 == 0 or count == total:
             elapsed = time.time() - start_time
             print(f"Translated {count}/{total} comments... ({elapsed:.2f} seconds elapsed)")

    comment_list = translated_comments
    print("Translation step complete.")

# --- Clean Text ---
if comment_list:
    print("Cleaning comments...")
    # Process cleaning in chunks potentially? For now, list comprehension
    comment_list = [clean_text(comment) for comment in comment_list if comment]
    comment_list = [comment for comment in comment_list if comment] # Remove empty strings
    print(f"Processing {len(comment_list)} non-empty comments after cleaning.")

# =========================================
# 6. Perform Analysis
# =========================================
# Performs Zero-Shot Classification for categories and overall Sentiment Analysis.

# --- Zero-Shot Classification for Categories ---
zsc_results = []
if comment_list:
    print(f"\nRunning zero-shot classification for categories on {len(comment_list)} comments...")
    candidates = CATEGORIES
    try:
        print(f"Attempting to run Zero-Shot on device: {'GPU 0' if target_device == 0 else 'CPU'}")
        classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device=target_device)
        # Batching might be less critical for 50 comments, but good practice
        # zsc_results = classifier(comment_list, candidate_labels=candidates, multi_label=False, batch_size=8)
        zsc_results = classifier(comment_list, candidate_labels=candidates, multi_label=False)
        print("Zero-shot classification complete.")
    except Exception as e:
        print(f"ERROR: Zero-shot classification failed. Details: {e}")
        zsc_results = [{"labels": ["Error"], "scores": [0.0]}] * len(comment_list) # Placeholder
else:
    print("Skipping Zero-Shot Classification as no comments available.")

# --- Sentiment Analysis (Overall Comment Sentiment) ---
sentiment_results = []
if comment_list:
    print(f"\nRunning sentiment analysis on {len(comment_list)} comments...")
    try:
        print(f"Attempting to run Sentiment Analysis on device: {'GPU 0' if target_device == 0 else 'CPU'}")
        sentiment_classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=target_device)
        # Batching might be less critical for 50 comments
        # sentiment_results = sentiment_classifier(comment_list, batch_size=8)
        sentiment_results = sentiment_classifier(comment_list)
        print("Sentiment analysis complete.")
    except Exception as e:
        print(f"ERROR: Sentiment analysis failed. Details: {e}")
        sentiment_results = [{"label": "Error", "score": 0.0}] * len(comment_list) # Placeholder
else:
    print("Skipping Sentiment Analysis as no comments available.")

# =========================================
# 7. Process Results & Generate Bar Charts
# =========================================
output_df = None # Initialize DataFrame
if comment_list and len(zsc_results) == len(comment_list) and len(sentiment_results) == len(comment_list):
    print("\nProcessing analysis results...")
    # Create DataFrame first
    try:
        output_df = pd.DataFrame({
            "comment": comment_list,
            "top_category": [res["labels"][0] for res in zsc_results],
            "category_score": [res["scores"][0] for res in zsc_results],
            "sentiment": [res["label"].upper() for res in sentiment_results], # Ensure consistent case
            "sentiment_score": [res["score"] for res in sentiment_results]
        })
    except Exception as e:
        print(f"ERROR: Failed to create results DataFrame. Details: {e}")
        output_df = None # Ensure df is None if creation fails

    if output_df is not None:
        # --- Generate Bar Chart of Negative Comments per Category ---
        print("\nGenerating bar chart for negative comments per category...")
        try:
            negative_comments_df = output_df[output_df['sentiment'] == 'NEGATIVE'].copy()
            if not negative_comments_df.empty:
                negative_counts = negative_comments_df['top_category'].value_counts().reindex(CATEGORIES, fill_value=0)
                plt.figure(figsize=(10, 7))
                bars = plt.bar(negative_counts.index, negative_counts.values, color='salmon')
                plt.xlabel("Category")
                plt.ylabel("Number of Negative Comments")
                # Update plot title
                plt.title(f"Negative Comments per Category (First {len(comment_list)} Processed Comments)")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                for bar in bars:
                    yval = bar.get_height()
                    if yval > 0: plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center')
                plt.savefig(OUTPUT_NEGATIVE_CHART_FILE)
                print(f"Negative comments chart saved to '{OUTPUT_NEGATIVE_CHART_FILE}' in your Google Drive.")
                # plt.show() # Commented out
                plt.close()
            else:
                print("No negative comments found to generate the chart.")
        except Exception as e:
            print(f"ERROR: Failed to generate or save the negative comments bar chart. Details: {e}")

        # --- Generate Bar Chart of Positive Comments per Category ---
        print("\nGenerating bar chart for positive comments per category...")
        try:
            positive_comments_df = output_df[output_df['sentiment'] == 'POSITIVE'].copy()
            if not positive_comments_df.empty:
                positive_counts = positive_comments_df['top_category'].value_counts().reindex(CATEGORIES, fill_value=0)
                plt.figure(figsize=(10, 7))
                bars = plt.bar(positive_counts.index, positive_counts.values, color='lightgreen') # Different color
                plt.xlabel("Category")
                plt.ylabel("Number of Positive Comments") # Updated label
                # Update plot title
                plt.title(f"Positive Comments per Category (First {len(comment_list)} Processed Comments)") # Updated title
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                for bar in bars:
                    yval = bar.get_height()
                    if yval > 0: plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center')
                plt.savefig(OUTPUT_POSITIVE_CHART_FILE) # Use new filename
                print(f"Positive comments chart saved to '{OUTPUT_POSITIVE_CHART_FILE}' in your Google Drive.")
                # plt.show() # Commented out
                plt.close()
            else:
                print("No positive comments found to generate the chart.")
        except Exception as e:
            print(f"ERROR: Failed to generate or save the positive comments bar chart. Details: {e}")

elif not comment_list:
     print("\nSkipping result processing and chart generation as no comments were processed.")
else:
    print(f"\nWarning: Mismatch in result lengths or missing analysis results. Cannot process results or generate charts.")
    print(f"Comments: {len(comment_list)}, ZSC results: {len(zsc_results)}, Sentiment results: {len(sentiment_results)}")


# =========================================
# 8. Save Detailed Results to Google Drive
# =========================================

if output_df is not None: # Check if DataFrame was created successfully
    print(f"\nSaving detailed results for the first {len(output_df)} processed comments to '{OUTPUT_CSV_FILE}'")
    try:
        # Save directly to the current directory (your Drive folder)
        output_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
        print(f"Saved {len(output_df)} classified comments to '{OUTPUT_CSV_FILE}' in {os.getcwd()}")
    except Exception as e:
        print(f"ERROR: Failed to save detailed results CSV. Details: {e}")
else:
    print("\nSkipping saving detailed results CSV as results were not processed or DataFrame creation failed.")


print("\nScript finished.")