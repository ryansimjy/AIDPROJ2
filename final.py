# -*- coding: utf-8 -*-
# Combined Product Review Analyzer Script
#
# Reads comments from a CSV, performs multi-stage analysis (Transformers, VADER, Gemini),
# visualizes results, and saves detailed output.
#
# Workflow: Load CSV -> Prepare Data -> ZSC (Transformers) -> Sentiment (Transformers) ->
#           Sentiment (VADER) -> Category/Sentiment (Gemini) -> Combine -> Visualize -> Save

# =========================================
# 1. Setup: Import Libraries
# =========================================
import pandas as pd
import os
import sys
import re
import ast # For safely evaluating string literals (like lists in CSV)
import datetime
import time
from collections import Counter

# NLP & Analysis
import nltk
try:
    # Try initializing VADER, handle download if necessary
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except (nltk.downloader.DownloadError, LookupError):
        print("Downloading VADER lexicon...")
        nltk.download('vader_lexicon', quiet=False)
    from nltk.sentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
    print("VADER Analyzer ready.")
except Exception as e:
    print(f"Warning: Could not initialize VADER Sentiment Analyzer. VADER scores will be skipped. Error: {e}")
    VADER_AVAILABLE = False
    SentimentIntensityAnalyzer = None # Define as None if unavailable

from deep_translator import GoogleTranslator
from transformers import pipeline, logging as hf_logging
import torch # To check GPU availability

# Google Gemini
from google import genai

# Visualization
import matplotlib.pyplot as plt

print("Core libraries imported.")

# =========================================
# 2. Configuration
# =========================================

# --- Input Data ---
# Recommendation: Use absolute paths for input files for clarity.
INPUT_CSV_FILE = "apple_vision_pro_comments.csv" # <<< CHANGE TO YOUR CSV FILE PATH (Absolute or relative to script)
COMMENT_COLUMN_NAME = "comments" # <<< CHANGE TO THE NAME OF THE COLUMN WITH COMMENTS
MAX_COMMENTS_TO_PROCESS = 1000 # Set to None to process all comments, or e.g., 100 for testing

# --- Analysis Categories ---
# Define categories for Zero-Shot Classification and Bar Chart
CATEGORIES = ["Price", "Hardware Quality", "Comfort", "Fit", "Eye Tracking", "Hand Tracking", "Passthrough", "Display Quality", "Field of View", "Software", "Apps", "Use Case", "Productivity", "Entertainment", "Battery life", "Connectivity", "Weight", "Setup", "Motion Sickness", "Material", "Headstrap"] # Combined list

# --- API Keys (IMPORTANT: Replace placeholders or use environment variables) ---
# WARNING: Hardcoding secrets is insecure. Consider environment variables or a config file.
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY" # <<< REPLACE WITH YOUR GEMINI API KEY

# --- Gemini Configuration ---
GEMINI_MODEL = "gemini-2.5-flash-preview-04-17"
GEMINI_BATCH_SIZE = 10000 # Number of comments to send to Gemini per API call (adjust based on limits/performance)
RUN_GEMINI_ANALYSIS = True

# --- Output Files ---
# Create a timestamped or specific output directory relative to the script's location
output_dir_name = f"analysis_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
# Use a base name for related output files
safe_input_name = "".join(c if c.isalnum() else "_" for c in os.path.splitext(os.path.basename(INPUT_CSV_FILE))[0]).strip('_')
BASE_OUTPUT_FILENAME = f"{safe_input_name}_analysis"
OUTPUT_CSV_FILE = f"{BASE_OUTPUT_FILENAME}_detailed.csv" # Will be saved inside output_dir_name
OUTPUT_CHART_FILE = f"{BASE_OUTPUT_FILENAME}_negative_categories.png" # Will be saved inside output_dir_name

# --- Environment & Execution ---
WORKING_DIR = None # Will be set below

# Check GPU availability for Transformers
if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    target_device = 0 # Use GPU
else:
    print("WARNING: No GPU detected by PyTorch. Transformer pipelines will run on CPU (might be slow).")
    target_device = -1 # Use CPU

# Optional: Suppress verbose logging from transformers if desired
# hf_logging.set_verbosity_error()

# =========================================
# 3. Environment Setup (Directory)
# =========================================

# For local execution, create output dir relative to the script's directory
script_location_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
WORKING_DIR = os.path.join(script_location_dir, output_dir_name)
os.makedirs(WORKING_DIR, exist_ok=True)
print(f"Running locally. Outputs will be saved to: {os.path.abspath(WORKING_DIR)}")

# Change to the working directory to simplify saving files
try:
    os.chdir(WORKING_DIR)
    print(f"Changed working directory to: {os.getcwd()}")
except Exception as e:
    print(f"ERROR: Could not change to working directory '{WORKING_DIR}'. Files may save elsewhere. Error: {e}")
    # Continue execution, but saving might fail or go to the original CWD

print("\n--- Configuration Summary ---")
print(f"Input CSV: {INPUT_CSV_FILE}")
print(f"Comment Column: {COMMENT_COLUMN_NAME}")
print(f"Max Comments: {'All' if MAX_COMMENTS_TO_PROCESS is None else MAX_COMMENTS_TO_PROCESS}")
print(f"Categories: {', '.join(CATEGORIES)}")
print(f"Run Gemini Analysis: {RUN_GEMINI_ANALYSIS}")
if RUN_GEMINI_ANALYSIS:
    print(f"Gemini Model: {GEMINI_MODEL}")
    print(f"Gemini Batch Size: {GEMINI_BATCH_SIZE}")
print(f"Output Directory: {os.getcwd()}") # Current working dir where files will be saved
print(f"Output CSV: {OUTPUT_CSV_FILE}")
print(f"Output Chart: {OUTPUT_CHART_FILE}")
print(f"Transformer Device: {'GPU 0' if target_device == 0 else 'CPU'}")
print("--- End Configuration ---")


# =========================================
# 4. Helper Functions
# =========================================

def safe_literal_eval(val):
    """Safely evaluate a string literal (like a list) or return list with the string."""
    if pd.isna(val):
        return []
    try:
        if isinstance(val, str):
            val_stripped = val.strip()
            if val_stripped.startswith('[') and val_stripped.endswith(']'):
                evaluated = ast.literal_eval(val_stripped)
                # Ensure result is a list, handle cases where eval might return non-list
                return evaluated if isinstance(evaluated, list) else [evaluated]
            else:
                 return [val] # Treat plain string as a single-item list
        elif isinstance(val, list):
            return val # Already a list
        else:
            # Try converting other types to string and wrap in list
            return [str(val)]
    except (ValueError, SyntaxError, TypeError, MemoryError):
        # If eval fails, return the original string wrapped in a list if possible
        return [str(val)] if isinstance(val, (str, int, float)) else []

def clean_text(text):
    """Cleans the input text for NLP tasks."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions (@user, /u/user, u/user)
    text = re.sub(r'[@/]?u/\w+', '', text)
    # Remove subreddit mentions (/r/sub, r/sub)
    text = re.sub(r'/?r/\w+', '', text)
    # Remove specific markdown/formatting if needed
    text = re.sub(r'\[.*?\]\(.*?\)', '', text) # Remove markdown links [text](url)
    text = re.sub(r'[`\*\_~]', '', text) # Remove backticks, asterisks, underscores, tildes
    # Remove special characters and numbers, keep spaces and basic punctuation if desired
    # Option 1: Keep only letters and spaces
    # text = re.sub(r'[^a-z\s]', '', text)
    # Option 2: Keep letters, numbers, spaces, and some basic punctuation
    text = re.sub(r'[^a-z0-9\s.,!?-]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("\nHelper functions defined.")

# =========================================
# 5. Load and Prepare Data
# =========================================

# --- Load comments from CSV ---
print(f"\nLoading comments from '{INPUT_CSV_FILE}'...")
df_input = None
all_comments_raw = []
original_indices = [] # To track mapping back to original DF rows

try:
    # Construct the full path if the provided path is relative
    csv_load_path = INPUT_CSV_FILE
    if not os.path.isabs(csv_load_path):
         # Assume it's relative to the script's original location
         script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
         csv_load_path = os.path.join(script_dir, INPUT_CSV_FILE)

    if not os.path.exists(csv_load_path):
         # If not found relative to script, maybe it's relative to where user ran python from?
         # Check the original INPUT_CSV_FILE path directly.
         if os.path.exists(INPUT_CSV_FILE):
             csv_load_path = INPUT_CSV_FILE
         else:
             raise FileNotFoundError(f"The file '{INPUT_CSV_FILE}' was not found either as an absolute path, relative to the script location ('{script_dir}'), or relative to the current execution directory.")

    df_input = pd.read_csv(csv_load_path)
    if COMMENT_COLUMN_NAME not in df_input.columns:
         raise ValueError(f"Column '{COMMENT_COLUMN_NAME}' not found in the CSV file.")
    print(f"Successfully loaded {len(df_input)} rows from {csv_load_path}.")

    # --- Extract and flatten comments ---
    print("Extracting and flattening comments...")
    for index, item in df_input[COMMENT_COLUMN_NAME].items(): # Use .items() to get index
        evaluated_comments = safe_literal_eval(item)
        for comment in evaluated_comments:
            comment_str = str(comment).strip()
            if comment_str: # Only add non-empty comments
                all_comments_raw.append(comment_str)
                original_indices.append(index) # Store the original row index

    print(f"Extracted {len(all_comments_raw)} non-empty comments.")

    # --- Limit comments if configured ---
    if MAX_COMMENTS_TO_PROCESS is not None and len(all_comments_raw) > MAX_COMMENTS_TO_PROCESS:
        print(f"Limiting processing to the first {MAX_COMMENTS_TO_PROCESS} comments.")
        comments_to_process_raw = all_comments_raw[:MAX_COMMENTS_TO_PROCESS]
        indices_to_process = original_indices[:MAX_COMMENTS_TO_PROCESS]
    else:
        comments_to_process_raw = all_comments_raw
        indices_to_process = original_indices
        print(f"Processing all {len(comments_to_process_raw)} extracted comments.")

    if not comments_to_process_raw:
        print("No comments found or extracted. Exiting.")
        sys.exit() # Exit if no comments to process

    # Create a DataFrame containing only the comments we will process and their original index
    df_processed_comments = pd.DataFrame({
        'original_comment': comments_to_process_raw,
        'original_df_index': indices_to_process
    })


except FileNotFoundError as e:
    print(f"ERROR: {e}")
    sys.exit()
except ValueError as e:
    print(f"ERROR: {e}")
    sys.exit()
except Exception as e:
    print(f"ERROR: Failed during data loading or initial processing. Details: {e}")
    sys.exit()


# --- Translate to English (Optional but Recommended for English models) ---
print(f"\nTranslating {len(df_processed_comments)} comments to English (if needed)... This may take time.")
translated_comments = []
count = 0
total = len(df_processed_comments)
start_time = time.time()
translator = GoogleTranslator(source="auto", target="en") # Initialize once

for i, comment in enumerate(df_processed_comments['original_comment']):
    if not comment.strip():
         translated_comments.append("")
         continue
    try:
        # Add a small delay if hitting rate limits (uncomment if needed)
        # time.sleep(0.05)
        translated = translator.translate(comment)
        translated_comments.append(translated if translated else comment) # Keep original if translation fails/is empty
    except Exception as e:
        # print(f"Warning: Translation failed for comment {i+1}. Keeping original. Error: {e}")
        translated_comments.append(comment) # Keep original on error
    count += 1
    # Print progress periodically
    if count % 100 == 0 or count == total:
         elapsed = time.time() - start_time
         print(f"  Translated {count}/{total} comments... ({elapsed:.2f} seconds elapsed)")

df_processed_comments['translated_comment'] = translated_comments
print("Translation step complete.")


# --- Clean Text ---
print("\nCleaning translated comments...")
df_processed_comments['cleaned_comment'] = df_processed_comments['translated_comment'].apply(clean_text)

# Filter out any comments that became empty after cleaning
non_empty_mask = df_processed_comments['cleaned_comment'] != ""
df_processed_comments = df_processed_comments[non_empty_mask].copy()
print(f"Retained {len(df_processed_comments)} non-empty comments after cleaning.")

if df_processed_comments.empty:
    print("No comments remained after cleaning. Exiting.")
    sys.exit()

# Prepare the list of cleaned comments for pipeline inputs
comment_list_cleaned = df_processed_comments['cleaned_comment'].tolist()

# =========================================
# 6. Perform Analysis (Transformers & VADER)
# =========================================

# --- Initialize VADER ---
vader_analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None

# --- Initialize Transformer Pipelines ---
zsc_classifier = None
sentiment_classifier = None
try:
    print(f"\nInitializing Zero-Shot Classification pipeline...")
    # Using a multilingual model can be robust even after translation
    zsc_classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        device=target_device
    )
    print(f"Initializing Sentiment Analysis pipeline...")
    sentiment_classifier = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=target_device
    )
    print("Transformer pipelines initialized.")
except Exception as e:
    print(f"ERROR: Failed to initialize Hugging Face pipelines. Details: {e}")
    # Decide if script should exit or continue without transformer analysis
    # sys.exit() # Or set flags to skip transformer steps

# --- Run Zero-Shot Classification ---
zsc_results = None
if zsc_classifier and comment_list_cleaned:
    print(f"\nRunning zero-shot classification for categories on {len(comment_list_cleaned)} comments...")
    try:
        # Process in batches for potentially large lists
        zsc_results = zsc_classifier(comment_list_cleaned, candidate_labels=CATEGORIES, multi_label=False, batch_size=16) # Adjust batch_size based on GPU memory
        print("Zero-shot classification complete.")
    except Exception as e:
        print(f"ERROR: Zero-shot classification failed. Details: {e}")
        # Create placeholder results
        zsc_results = [{"labels": ["Error"], "scores": [0.0]}] * len(comment_list_cleaned)
else:
     print("Skipping Zero-Shot Classification (pipeline not initialized or no comments).")

# --- Run Transformer Sentiment Analysis ---
sentiment_results = None
if sentiment_classifier and comment_list_cleaned:
    print(f"\nRunning Transformer sentiment analysis on {len(comment_list_cleaned)} comments...")
    try:
        # Process in batches
        sentiment_results = sentiment_classifier(comment_list_cleaned, batch_size=16) # Adjust batch_size
        print("Transformer sentiment analysis complete.")
    except Exception as e:
        print(f"ERROR: Transformer sentiment analysis failed. Details: {e}")
        # Create placeholder results
        sentiment_results = [{"label": "Error", "score": 0.0}] * len(comment_list_cleaned)
else:
    print("Skipping Transformer Sentiment Analysis (pipeline not initialized or no comments).")

# --- Run VADER Sentiment Analysis ---
vader_scores = []
if vader_analyzer and comment_list_cleaned:
    print(f"\nRunning VADER sentiment analysis on {len(comment_list_cleaned)} comments...")
    try:
        vader_scores = [vader_analyzer.polarity_scores(comment)['compound'] for comment in comment_list_cleaned]
        print("VADER sentiment analysis complete.")
    except Exception as e:
        print(f"ERROR: VADER sentiment analysis failed. Details: {e}")
        vader_scores = [0.0] * len(comment_list_cleaned) # Placeholder
else:
    print("Skipping VADER Sentiment Analysis (analyzer not available or no comments).")
    vader_scores = [None] * len(comment_list_cleaned) # Use None if skipped


# --- Add Transformer & VADER results to DataFrame ---
if zsc_results and len(zsc_results) == len(df_processed_comments):
    df_processed_comments['zsc_category'] = [res.get("labels", ["N/A"])[0] for res in zsc_results]
    df_processed_comments['zsc_category_score'] = [res.get("scores", [0.0])[0] for res in zsc_results]
else:
    df_processed_comments['zsc_category'] = "Error/Skipped"
    df_processed_comments['zsc_category_score'] = 0.0

if sentiment_results and len(sentiment_results) == len(df_processed_comments):
    df_processed_comments['transformer_sentiment'] = [res.get("label", "Error").upper() for res in sentiment_results]
    df_processed_comments['transformer_sentiment_score'] = [res.get("score", 0.0) for res in sentiment_results]
else:
    df_processed_comments['transformer_sentiment'] = "Error/Skipped"
    df_processed_comments['transformer_sentiment_score'] = 0.0

if vader_scores and len(vader_scores) == len(df_processed_comments):
     df_processed_comments['vader_compound_score'] = vader_scores
else:
     df_processed_comments['vader_compound_score'] = None


# =========================================
# 7. Perform Analysis (Gemini)
# =========================================
gemini_results_list = [] # Store dictionaries {'gemini_category': ..., 'gemini_sentiment': ...}

if RUN_GEMINI_ANALYSIS and genai and comment_list_cleaned:
    print(f"\n--- Starting Gemini Analysis ({GEMINI_MODEL}) ---")
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY" or not GEMINI_API_KEY:
        print("ERROR: Gemini API Key not configured. Skipping Gemini analysis.")
        RUN_GEMINI_ANALYSIS = False
    else:
        try:
            client = genai.Client(api_key="GEMINI_API_KEY")

            print(f"Gemini client configured for model: {GEMINI_MODEL}")

            num_comments = len(comment_list_cleaned)
            gemini_raw_responses = {} # Store raw text response per comment index

            # Process comments in batches
            for i in range(0, num_comments, GEMINI_BATCH_SIZE):
                batch_comments = comment_list_cleaned[i : i + GEMINI_BATCH_SIZE]
                batch_indices = list(range(i, i + len(batch_comments))) # Original indices within comment_list_cleaned

                print(f"\nProcessing Gemini batch {i // GEMINI_BATCH_SIZE + 1} ({len(batch_comments)} comments)...")

                # Format comments with index numbers for the prompt
                formatted_comments_for_prompt = "\n".join([f"{idx+1}. \"{comment}\"" for idx, comment in zip(batch_indices, batch_comments)])
                categories_str = ", ".join(CATEGORIES)

                # Construct the prompt (similar to test.py)
                prompt = f"""Analyze each of the following comments, numbered according to their original index. For each comment, determine if it primarily discusses ONE of the following categories: {categories_str}.

If a comment fits into ONE of these categories, identify that single category and determine if the sentiment expressed towards that topic is Positive or Negative.
If a comment does not clearly fit into any of these specific categories, or discusses multiple categories, output "Category: N/A, Sentiment: N/A".

Return the analysis for each comment on a new line, starting with the original comment index number, followed by the category and sentiment.

Comments to Analyze:
{formatted_comments_for_prompt}

Desired Output Format (one line per comment, using the original index number):
[Original Index Number + 1]. Category: [Category Name], Sentiment: [Positive/Negative]
OR
[Original Index Number + 1]. Category: N/A, Sentiment: N/A

Example Output:
1. Category: Price, Sentiment: Negative
2. Category: Comfort, Sentiment: Positive
5. Category: N/A, Sentiment: N/A
...
{batch_indices[-1]+1}. Category: Battery life, Sentiment: Negative

Provide ONLY the results in the specified format, starting from index {batch_indices[0]+1}, without any introductory text, concluding remarks, or other explanations."""

                try:
                    start_time_batch = time.time()
                    # print(f"DEBUG: Sending prompt to Gemini:\n---\n{prompt}\n---") # Uncomment for debugging
                    response = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=[prompt]
                    )
                    elapsed_batch = time.time() - start_time_batch
                    print(f"  Gemini batch response received in {elapsed_batch:.2f} seconds.")
                    # print(f"DEBUG: Gemini Raw Response:\n---\n{response.text}\n---") # Uncomment for debugging

                    # Process the response text for this batch
                    for line in response.text.splitlines():
                        line = line.strip()
                        if not line: continue

                        # Try to parse the line based on the expected format "Index. Category: X, Sentiment: Y"
                        match = re.match(r"(\d+)\.\s*Category:\s*(.+?),\s*Sentiment:\s*(.+)", line, re.IGNORECASE)
                        if match:
                            try:
                                index = int(match.group(1)) - 1 # Convert back to 0-based index
                                category = match.group(2).strip()
                                sentiment = match.group(3).strip().upper() # Standardize case

                                # Validate category and sentiment
                                if category not in CATEGORIES and category != "N/A":
                                    print(f"  Warning: Gemini returned unknown category '{category}' for index {index}. Treating as N/A.")
                                    category = "N/A"
                                if sentiment not in ["POSITIVE", "NEGATIVE", "N/A"]:
                                    print(f"  Warning: Gemini returned unknown sentiment '{sentiment}' for index {index}. Treating as N/A.")
                                    sentiment = "N/A"

                                # Ensure index is within the bounds of the current batch processing
                                if i <= index < i + GEMINI_BATCH_SIZE:
                                     gemini_raw_responses[index] = {'gemini_category': category, 'gemini_sentiment': sentiment}
                                else:
                                     print(f"  Warning: Gemini returned index {index+1} which is outside the current batch range ({i+1} to {i+GEMINI_BATCH_SIZE}). Skipping.")

                            except (ValueError, IndexError) as parse_err:
                                print(f"  Warning: Could not parse Gemini response line: '{line}'. Error: {parse_err}")
                        else:
                            print(f"  Warning: Gemini response line did not match expected format: '{line}'")

                except Exception as api_err:
                    print(f"ERROR: Gemini API call failed for batch starting at index {i}. Details: {api_err}")
                    # Fill results for this batch with errors
                    for idx in batch_indices:
                        if idx not in gemini_raw_responses: # Avoid overwriting if partial success
                             gemini_raw_responses[idx] = {'gemini_category': 'API Error', 'gemini_sentiment': 'API Error'}
                    # Optional: break or sleep before retrying/next batch
                    time.sleep(5) # Wait before next batch on error

            # --- Integrate Gemini results into the DataFrame ---
            # Create lists based on the order of df_processed_comments
            gemini_categories = [gemini_raw_responses.get(i, {'gemini_category': 'Not Processed'})['gemini_category'] for i in range(num_comments)]
            gemini_sentiments = [gemini_raw_responses.get(i, {'gemini_sentiment': 'Not Processed'})['gemini_sentiment'] for i in range(num_comments)]

            if len(gemini_categories) == len(df_processed_comments):
                df_processed_comments['gemini_category'] = gemini_categories
                df_processed_comments['gemini_sentiment'] = gemini_sentiments
                print("Gemini analysis results integrated into DataFrame.")
            else:
                 print(f"ERROR: Mismatch between number of comments ({len(df_processed_comments)}) and Gemini results ({len(gemini_categories)}). Skipping Gemini column add.")
                 df_processed_comments['gemini_category'] = 'Processing Error'
                 df_processed_comments['gemini_sentiment'] = 'Processing Error'


        except Exception as general_gemini_err:
            print(f"ERROR: An unexpected error occurred during Gemini setup or processing. Details: {general_gemini_err}")
            df_processed_comments['gemini_category'] = 'Setup Error'
            df_processed_comments['gemini_sentiment'] = 'Setup Error'
            RUN_GEMINI_ANALYSIS = False # Disable further Gemini steps if setup fails

else:
    print("\nSkipping Gemini Analysis (disabled, library unavailable, or API key missing).")
    df_processed_comments['gemini_category'] = 'Skipped'
    df_processed_comments['gemini_sentiment'] = 'Skipped'


# =========================================
# 8. Combine Final Results & Visualize
# =========================================
analysis_df = df_processed_comments # Use the DataFrame we've been adding columns to

print("\n--- Final Analysis DataFrame ---")
print(analysis_df.head())
print(f"DataFrame shape: {analysis_df.shape}")
# Display value counts for key columns
print("\nTransformer Sentiment Distribution:")
print(analysis_df['transformer_sentiment'].value_counts())
print("\nZSC Category Distribution (Top 10):")
print(analysis_df['zsc_category'].value_counts().head(10))
if RUN_GEMINI_ANALYSIS and 'gemini_sentiment' in analysis_df.columns:
    print("\nGemini Sentiment Distribution:")
    print(analysis_df['gemini_sentiment'].value_counts())
    print("\nGemini Category Distribution (Top 10):")
    print(analysis_df['gemini_category'].value_counts().head(10))


# --- Generate Bar Chart of Negative Comments per Category ---
# Based on Transformer Sentiment and ZSC Category for consistency with original scripts
print("\nGenerating bar chart for negative comments per category (using Transformer results)...")
try:
    # Filter for negative comments based on the TRANSFORMER's output
    negative_comments_df = analysis_df[analysis_df['transformer_sentiment'] == 'NEGATIVE'].copy()

    if not negative_comments_df.empty:
        # Count occurrences of each ZSC category within negative comments
        negative_counts = negative_comments_df['zsc_category'].value_counts()

        # Ensure all defined categories are included, even if count is 0, and filter out 'Error/Skipped' etc.
        valid_categories = [cat for cat in CATEGORIES if cat in negative_counts.index] # Only include categories that actually appeared
        negative_counts = negative_counts.reindex(valid_categories, fill_value=0)

        # Filter out categories with 0 counts for cleaner chart
        negative_counts = negative_counts[negative_counts > 0].sort_values(ascending=False)

        if not negative_counts.empty:
             plt.figure(figsize=(12, 8)) # Adjust size as needed
             bars = plt.bar(negative_counts.index, negative_counts.values, color='salmon')
             plt.xlabel("Category (from Zero-Shot Classification)")
             plt.ylabel("Number of Negative Comments (from Transformer Sentiment)")
             plt.title(f"Negative Comments per Category ({len(negative_comments_df)} Neg Comments Found)")
             plt.xticks(rotation=45, ha='right') # Rotate labels for readability
             plt.tight_layout() # Adjust layout

             # Add counts on top of bars
             for bar in bars:
                 yval = bar.get_height()
                 if yval > 0:
                     plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center')

             # Save the chart inside the WORKING_DIR (which is the current directory)
             plt.savefig(OUTPUT_CHART_FILE)
             print(f"Bar chart saved to: {OUTPUT_CHART_FILE}")
             # plt.show() # Optionally display inline if not running headless
             plt.close() # Close plot to free memory
        else:
             print("No negative comments found for the specified valid categories after filtering.")

    else:
        print("No comments classified as NEGATIVE by the Transformer sentiment pipeline.")

except Exception as e:
    print(f"ERROR: Failed to generate or save the bar chart. Details: {e}")


# =========================================
# 9. Save Detailed Analysis Results
# =========================================

if not analysis_df.empty:
    print(f"\nSaving detailed analysis results for {len(analysis_df)} comments to '{OUTPUT_CSV_FILE}'...")
    try:
        # Save to the current working directory (set earlier)
        analysis_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig') # utf-8-sig for better Excel compatibility
        print(f"Saved detailed analysis CSV to: {os.path.join(os.getcwd(), OUTPUT_CSV_FILE)}")
    except Exception as e:
        print(f"ERROR: Failed to save detailed analysis CSV. Details: {e}")
else:
    print("\nSkipping saving detailed analysis CSV as the DataFrame is empty or was not generated.")

# =========================================
# 10. Conclusion
# =========================================
print("\nScript finished.")