# -*- coding: utf-8 -*-
# # Reddit Product Review Analyzer (Based on YouTube Analysis Structure) - GPU Enabled
#
# This script analyzes Reddit comments about a specific product scraped using PRAW.
# It utilizes the provided `RedditScraper` class.
#
# **Functionality:**
# 1.  **Setup:** Installs libraries, imports modules, handles optional Google Drive mounting (for Colab).
# 2.  **Configuration:** Sets Reddit API credentials, product name, subreddits, categories, etc.
# 3.  **Scraping:** Uses the `RedditScraper` class to fetch posts and comments.
# 4.  **Prepare Data:** Extracts comments, translates (optional but included), cleans text.
# 5.  **Analysis (GPU):** Performs Zero-Shot Classification for categories and Sentiment Analysis (using Transformers) for positive/negative labels.
# 6.  **Visualize & Save:** Generates a bar chart of negative comments per category, saves the chart and detailed analysis results (CSV) to a specified directory (or Google Drive).
#
# **Prerequisites:**
# * **IMPORTANT:** If using Colab, ensure you have selected a GPU runtime (`Runtime` -> `Change runtime type` -> `GPU`).
# * Run the `!pip install` command below if needed.
# * Update Reddit API credentials and user agent string in the Configuration section.
# * If using Colab & Google Drive, ensure the target directory exists.

# =========================================
# 1. Setup: Install Libraries
# =========================================
# Note: praw is needed for the scraper class, sentencepiece for some transformer models
# !pip install pandas praw deep_translator transformers torch matplotlib nltk sentencepiece --quiet
# print("Libraries installed or checked.")

# =========================================
# 1. Setup: Import Libraries & Scraper Class
# =========================================
import praw
import pandas as pd
import datetime
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer # Keep for the scraper's initial pass
from prawcore.exceptions import PrawcoreException, NotFound
import time
import re
from deep_translator import GoogleTranslator
from transformers import pipeline, logging as hf_logging
import matplotlib.pyplot as plt
from collections import Counter
import torch # To check GPU availability

# Attempt Google Drive integration for Colab
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print("Core libraries imported.")

# --- Paste or Import your RedditScraper Class Here ---
# Ensure the class definition provided in the prompt is included here
class RedditScraper:
    """
    A class to scrape Reddit for product reviews within specified subreddits,
    perform sentiment analysis, and save the results. Operates in read-only mode.
    Handles subreddit names or URLs.
    (Class definition as provided in the prompt)
    """
    def __init__(self, client_id: str | None, client_secret: str | None, user_agent: str):
        """ Initializes the scraper and authenticates with Reddit in read-only mode. """
        self.reddit = self._authenticate(client_id, client_secret, user_agent)
        if self.reddit:
            try:
                print(f"Authentication successful (read-only mode). User Agent: {user_agent}")
                # Try initializing VADER, handle download if necessary
                try:
                    nltk.data.find('sentiment/vader_lexicon.zip')
                except (nltk.downloader.DownloadError, LookupError):
                    print("Downloading VADER lexicon for initial sentiment analysis...")
                    nltk.download('vader_lexicon', quiet=False)
                self.sia = SentimentIntensityAnalyzer()
            except Exception as e:
                print(f"An unexpected error occurred during VADER initialization: {e}")
                self.reddit = None # Consider authentication failed if VADER setup fails
                self.sia = None
        else:
            self.sia = None # Ensure sia is None if authentication failed

    def _authenticate(self, client_id: str | None, client_secret: str | None, user_agent: str) -> praw.Reddit | None:
        """ Handles Reddit authentication using PRAW in read-only mode. """
        if not all([client_id, client_secret]):
            print("Authentication failed: Client ID or Client Secret provided to the class is missing.")
            return None
        if not user_agent:
             print("Authentication failed: User Agent string is missing or empty.")
             return None
        try:
            print("Attempting to authenticate with Reddit (read-only mode)...")
            reddit_instance = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            # Test connection by trying to fetch a default subreddit listing (read-only action)
            list(reddit_instance.subreddits.default(limit=1)) # Use list() to force execution
            print("Read-only connection test successful.")
            return reddit_instance
        except PrawcoreException as e:
            print(f"Authentication failed: {e}")
            print("Common causes: Invalid Client ID/Secret, incorrect app type (script vs webapp), API rate limits, incorrect User Agent format.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during authentication: {e}")
            return None

    def _extract_subreddit_name(self, url_or_name: str) -> str | None:
        """ Extracts the subreddit name from a URL or returns the name if it's not a URL. """
        url_or_name = url_or_name.strip().lower() # Lowercase early
        # Improved regex to handle various reddit URL formats
        match = re.search(r'(?:www\.|old\.)?reddit\.com/r/([^/]+)', url_or_name)
        if match: return match.group(1)
        # Handle just 'r/subreddit' or 'subreddit'
        match = re.search(r'^(?:r/)?([^/]+)', url_or_name)
        if match and '/' not in match.group(1) and '.' not in match.group(1): # Avoid matching invalid names
            return match.group(1)
        print(f"  Warning: Could not extract a valid subreddit name from input: '{url_or_name}'")
        return None

    def search_reviews(self, product_name: str, subreddits_or_posts: list[str], limit_per_source: int = 100, fetch_comments: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Searches specified subreddits for posts OR directly fetches comments from specific post URLs.
        Also fetches comments from found posts if fetch_comments is True.
        """
        if not self.reddit or not self.sia:
            print("Scraper not properly initialized or authentication failed. Cannot search.")
            return pd.DataFrame(), pd.DataFrame()
        if not subreddits_or_posts:
            print("Error: No subreddits or post URLs provided to search in.")
            return pd.DataFrame(), pd.DataFrame()

        posts_data, comments_data = [], []
        processed_post_ids = set() # Track posts processed to avoid duplicates if a URL is also found via search

        print(f"\n--- Starting data retrieval for '{product_name}' ---")
        for source_input in subreddits_or_posts:
            is_post_url = '/comments/' in source_input # Check if it's a post URL
            sub_name = None if is_post_url else self._extract_subreddit_name(source_input)

            if is_post_url:
                post_id = re.search(r'/comments/([^/]+)/', source_input)
                if post_id:
                    post_id = post_id.group(1)
                    print(f"Processing specific post URL: {source_input} (ID: {post_id})")
                    if post_id in processed_post_ids:
                        print(f"  Skipping post {post_id}, already processed.")
                        continue
                    try:
                        post = self.reddit.submission(id=post_id)
                        post.title # Access an attribute to check if fetch was successful
                        post_entry = self._process_post(post)
                        posts_data.append(post_entry)
                        processed_post_ids.add(post.id)
                        if fetch_comments:
                            self._fetch_and_process_comments(post, comments_data, f"post {post.id}")
                    except NotFound: print(f"  Warning: Post {post_id} not found. Skipping.")
                    except PrawcoreException as e: print(f"  Error fetching post {post_id}: {e}. Skipping.")
                    except Exception as e: print(f"  An unexpected error processing post {post_id}: {e}. Skipping.")
                else:
                    print(f"  Warning: Could not extract post ID from URL: '{source_input}'. Skipping.")

            elif sub_name: # It's a subreddit name/URL
                print(f"Processing subreddit: r/{sub_name} (from input: '{source_input}')")
                try:
                    subreddit = self.reddit.subreddit(sub_name)
                    subreddit.display_name # Check existence
                    # Search for the product name within the subreddit
                    query = f'"{product_name}"'
                    print(f"  Searching r/{sub_name} for query: {query} (limit: {limit_per_source})")
                    results = subreddit.search(query, limit=limit_per_source, sort='relevance') # 'relevance' or 'new' or 'comments'
                    count = 0
                    for post in results:
                        if post.id in processed_post_ids: continue # Avoid duplicates if found via search
                        post_entry = self._process_post(post)
                        posts_data.append(post_entry)
                        processed_post_ids.add(post.id)
                        if fetch_comments:
                             self._fetch_and_process_comments(post, comments_data, f"r/{sub_name}")
                        count += 1
                        if count % 25 == 0: print(f"  Processed {count} posts from r/{sub_name} search...")
                    print(f"Finished searching r/{sub_name}. Found {count} relevant posts.")
                except NotFound: print(f"  Warning: Subreddit r/{sub_name} not found or access restricted. Skipping.")
                except PrawcoreException as e: print(f"  Error searching r/{sub_name}: {e}. Skipping.")
                except Exception as e: print(f"  An unexpected error while searching r/{sub_name}: {e}. Skipping.")
            else:
                 print(f"Skipping invalid input: '{source_input}'")

        print(f"--- Data retrieval complete. Found {len(posts_data)} posts and {len(comments_data)} comments. ---")
        return pd.DataFrame(posts_data) if posts_data else pd.DataFrame(), pd.DataFrame(comments_data) if comments_data else pd.DataFrame()

    def _fetch_and_process_comments(self, post: praw.models.Submission, comments_data_list: list, source_desc: str):
        """ Fetches and processes comments for a given post, adding them to the list. """
        print(f"    Fetching comments for post '{post.id}' ({source_desc})...")
        comment_count = 0
        try:
            post.comments.replace_more(limit=None) # Fetch all comments, might take time/resources
            for comment in post.comments.list():
                if isinstance(comment, praw.models.Comment):
                    comments_data_list.append(self._process_comment(comment, post.id))
                    comment_count += 1
            print(f"    Fetched {comment_count} comments for post '{post.id}'.")
        except PrawcoreException as e: print(f"    Warning: Could not fetch/process comments for post {post.id} ({source_desc}): {e}")
        except Exception as e: print(f"    Warning: An unexpected error processing comments for post {post.id} ({source_desc}): {e}")


    def _process_post(self, post: praw.models.Submission) -> dict:
        """ Extracts relevant data and VADER sentiment from a Reddit post. """
        text_content = (post.title + " " + post.selftext).strip()
        vader_score = self.sia.polarity_scores(text_content)['compound'] if text_content else 0.0
        author_name = str(post.author) if post.author else "[deleted]"
        return {
            'post_id': post.id,
            'subreddit': post.subreddit.display_name,
            'author': author_name,
            'title': post.title,
            'content': post.selftext,
            'url': f"https://www.reddit.com{post.permalink}",
            'score': post.score,
            'upvote_ratio': post.upvote_ratio,
            'num_comments': post.num_comments,
            'created_utc': datetime.datetime.fromtimestamp(post.created_utc, datetime.timezone.utc),
            'vader_sentiment': vader_score # Store VADER score
        }

    def _process_comment(self, comment: praw.models.Comment, post_id: str) -> dict:
        """ Extracts relevant data and VADER sentiment from a Reddit comment. """
        vader_score = self.sia.polarity_scores(comment.body)['compound'] if comment.body else 0.0
        author_name = str(comment.author) if comment.author else "[deleted]"
        return {
            'comment_id': comment.id,
            'post_id': post_id,
            'author': author_name,
            'content': comment.body,
            'url': f"https://www.reddit.com{comment.permalink}",
            'score': comment.score,
            'created_utc': datetime.datetime.fromtimestamp(comment.created_utc, datetime.timezone.utc),
            'vader_sentiment': vader_score # Store VADER score
        }

print("RedditScraper class defined.")


# =========================================
# 1. Setup: Environment Checks (GPU, Drive)
# =========================================

# Check GPU availability
if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    target_device = 0 # Use GPU
else:
    print("WARNING: No GPU detected by PyTorch. Pipelines will run on CPU (might be slow).")
    target_device = -1 # Use CPU

# Optional: Suppress verbose logging from transformers if desired
# hf_logging.set_verbosity_error()

# Google Drive Mounting (Colab specific)
DRIVE_MOUNT_PATH = '/content/drive'
PROJECT_DIR_IN_DRIVE = '/MyDrive/reddit_analysis_output' # CHANGE if needed
WORKING_DIR = None

if IN_COLAB:
    try:
        drive.mount(DRIVE_MOUNT_PATH, force_remount=True)
        print("Google Drive mounted successfully.")
        WORKING_DIR = os.path.join(DRIVE_MOUNT_PATH, PROJECT_DIR_IN_DRIVE.lstrip('/'))
        os.makedirs(WORKING_DIR, exist_ok=True)
        os.chdir(WORKING_DIR)
        print(f"Changed working directory to: {os.getcwd()}")
    except Exception as e:
        print(f"ERROR: Failed to mount Drive or set working directory. {e}")
        print(f"Will save outputs locally in Colab environment if possible.")
        WORKING_DIR = "/content/reddit_analysis_output" # Fallback local dir
        os.makedirs(WORKING_DIR, exist_ok=True)
        os.chdir(WORKING_DIR)
else:
    # For local execution, use a relative path
    WORKING_DIR = "reddit_analysis_output"
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.chdir(WORKING_DIR) # Change to output dir for saving files easily
    print(f"Running locally. Set working directory to: {os.getcwd()}")

# =========================================
# 2. Configuration
# =========================================

# --- Reddit API Credentials ---
# WARNING: Hardcoding secrets is insecure. Consider environment variables or config files.
# --- >>> REPLACE THESE WITH YOUR ACTUAL, CURRENT CREDENTIALS <<< ---
HARDCODED_CLIENT_ID = 'CLIENT_ID' # Replace with your Client ID
HARDCODED_CLIENT_SECRET = 'SECRET' # Replace with your Secret (RESET RECOMMENDED!)
# --- >>> END OF HARDCODED CREDENTIALS <<< ---

# --- Reddit User Agent ---
# REQUIRED by Reddit API rules. Replace 'YourRedditUsername'.
REDDIT_USERNAME_FOR_AGENT = "USERNAME" # <--- !!! REPLACE with your Reddit username !!!
REDDIT_USER_AGENT = f"script:product_review_analyzer_gpu:v1.0 (by u/{REDDIT_USERNAME_FOR_AGENT})"

# --- Search & Analysis Parameters ---
PRODUCT_TO_SEARCH = "Apple Vision Pro"  # <--- CHANGE AS NEEDED
# List of subreddit names, subreddit URLs, or specific post URLs
# The scraper will search subreddits for the product name
# AND fetch comments directly from post URLs.
SOURCES_TO_PROCESS = [                     # <--- CHANGE AS NEEDED
    "apple", # Subreddit name
    "virtualreality", # Subreddit name
    "VisionPro", # Subreddit name
    "https://www.reddit.com/r/apple/comments/1ai5xfo/the_thing_no_one_will_say_about_apple_vision_pro/", # Specific Post URL
    "https://www.reddit.com/r/VisionPro/comments/1ahvgth/average_persons_review_of_the_vision_pro/", # Specific Post URL
    # Add more subreddits or post URLs here
    # Example: "https://www.reddit.com/r/gadgets/"
]
SEARCH_LIMIT_PER_SUB = 50 # Max posts to check per subreddit search query
FETCH_ALL_COMMENTS = True # Set to True to get all comments from posts (can be slow), False for top-level only (faster)


# --- Analysis Categories ---
# Define categories for Zero-Shot Classification and Bar Chart
CATEGORIES = ["Price", "Hardware Quality", "Comfort", "Fit", "Eye Tracking", "Hand Tracking", "Passthrough", "Display Quality", "Field of View", "Software", "Apps", "Use Case", "Productivity", "Entertainment", "Battery life", "Connectivity", "Weight", "Setup", "Motion Sickness"]

# --- Output Files ---
# Timestamps will be added automatically later
safe_product_name = "".join(c if c.isalnum() else "_" for c in PRODUCT_TO_SEARCH).strip('_')
BASE_OUTPUT_FILENAME = f"reddit_analysis_{safe_product_name}"
OUTPUT_CSV_FILE = f"{BASE_OUTPUT_FILENAME}_detailed.csv"
OUTPUT_CHART_FILE = f"{BASE_OUTPUT_FILENAME}_negative_categories.png"
OUTPUT_POSTS_FILE = f"{BASE_OUTPUT_FILENAME}_posts_scraped.csv" # To save the posts found
OUTPUT_COMMENTS_RAW_FILE = f"{BASE_OUTPUT_FILENAME}_comments_scraped.csv" # To save raw comments

print("\nConfiguration set:")
print(f" - Product: {PRODUCT_TO_SEARCH}")
print(f" - Sources: {len(SOURCES_TO_PROCESS)} sources configured")
print(f" - Categories: {', '.join(CATEGORIES)}")
print(f" - Output Base Name: {BASE_OUTPUT_FILENAME}")
print(f" - Working Directory (for outputs): {os.getcwd()}")


# =========================================
# 3. Reddit Scraping
# =========================================
print("\nInitializing Reddit Scraper...")
scraper = RedditScraper(
    client_id=HARDCODED_CLIENT_ID,
    client_secret=HARDCODED_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

posts_df = pd.DataFrame()
comments_df = pd.DataFrame()

if scraper.reddit: # Proceed only if authentication was successful
    print("Starting Reddit data scraping...")
    posts_df, comments_df = scraper.search_reviews(
        product_name=PRODUCT_TO_SEARCH,
        subreddits_or_posts=SOURCES_TO_PROCESS,
        limit_per_source=SEARCH_LIMIT_PER_SUB,
        fetch_comments=FETCH_ALL_COMMENTS # Controls comment fetching depth
        )

    # --- Save Raw Scraped Data ---
    if not posts_df.empty:
        try:
            posts_df.to_csv(OUTPUT_POSTS_FILE, index=False, encoding='utf-8-sig')
            print(f"Saved {len(posts_df)} scraped posts to: {OUTPUT_POSTS_FILE}")
        except Exception as e:
            print(f"Error saving posts CSV: {e}")
    else:
        print("No post data scraped.")

    if not comments_df.empty:
        try:
            comments_df.to_csv(OUTPUT_COMMENTS_RAW_FILE, index=False, encoding='utf-8-sig')
            print(f"Saved {len(comments_df)} scraped comments to: {OUTPUT_COMMENTS_RAW_FILE}")
        except Exception as e:
            print(f"Error saving raw comments CSV: {e}")
    else:
        print("No comment data scraped.")

else:
    print("\nSkipping scraping due to authentication failure or initialization error.")


# =========================================
# 4. Define Helper Functions (Cleaning) & Prepare Data
# =========================================

def clean_text(text):
    """Cleans the input text for NLP tasks."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions
    text = re.sub(r'/u/\w+|u/\w+', '', text)
    # Remove subreddit mentions
    text = re.sub(r'/r/\w+|r/\w+', '', text)
    # Remove special characters and numbers, keep spaces
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("\nHelper functions defined.")

# --- Prepare Comment List for Analysis ---
comment_list_raw = []
if not comments_df.empty and 'content' in comments_df.columns:
    comment_list_raw = comments_df['content'].dropna().astype(str).tolist()
    print(f"\nExtracted {len(comment_list_raw)} non-null comments for analysis.")
else:
    print("\nNo comments found in DataFrame or 'content' column missing. Skipping analysis.")

# --- Translate to English (Optional but Recommended) ---
comment_list_translated = []
if comment_list_raw:
    print(f"Translating {len(comment_list_raw)} comments to English (if needed)... This may take time.")
    count = 0
    total = len(comment_list_raw)
    start_time = time.time()
    translator = GoogleTranslator(source="auto", target="en") # Initialize once
    for i, comment in enumerate(comment_list_raw):
        if not comment.strip(): # Skip empty comments
             comment_list_translated.append("")
             continue
        try:
            # Add a small delay to potentially avoid rate limits
            # time.sleep(0.05)
            translated = translator.translate(comment)
            comment_list_translated.append(translated if translated else comment) # Keep original if translation fails/is empty
        except Exception as e:
            # print(f"Warning: Translation failed for comment {i+1}. Keeping original. Error: {e}")
            comment_list_translated.append(comment) # Keep original on error
        count += 1
        # Print progress periodically
        if count % 100 == 0 or count == total:
             elapsed = time.time() - start_time
             print(f"  Translated {count}/{total} comments... ({elapsed:.2f} seconds elapsed)")

    print("Translation step complete.")
else:
    print("Skipping translation as no comments were extracted.")


# --- Clean Text ---
comment_list_cleaned = []
if comment_list_translated:
    print("Cleaning comments...")
    comment_list_cleaned = [clean_text(comment) for comment in comment_list_translated]
    # Filter out any comments that become empty after cleaning
    original_indices = [i for i, comment in enumerate(comment_list_cleaned) if comment]
    comment_list_cleaned = [comment for comment in comment_list_cleaned if comment]
    print(f"Processing {len(comment_list_cleaned)} non-empty comments after cleaning.")

    # IMPORTANT: Keep track of which original comments survived cleaning
    if len(original_indices) != len(comment_list_cleaned):
         print("Warning: Some comments became empty after cleaning and were removed.")
         # Filter the original comments_df to align with cleaned comments
         comments_df_filtered = comments_df.iloc[original_indices].copy()
         print(f"Filtered DataFrame down to {len(comments_df_filtered)} rows matching cleaned comments.")
    else:
         comments_df_filtered = comments_df.copy() # All comments survived
         print("All comments remained non-empty after cleaning.")

else:
    print("Skipping cleaning as no comments were translated/available.")
    comments_df_filtered = pd.DataFrame() # Ensure it's an empty DF


# =========================================
# 5. Perform Analysis (Pipelines)
# =========================================
# Uses the cleaned comment list

zsc_results = None
sentiment_results = None

if comment_list_cleaned:
    # --- Zero-Shot Classification for Categories ---
    print(f"\nRunning zero-shot classification for categories on {len(comment_list_cleaned)} comments...")
    try:
        print(f"Attempting to run Zero-Shot on device: {'GPU 0' if target_device == 0 else 'CPU'}")
        # Using a multilingual model might be beneficial if translation wasn't perfect
        # Consider "facebook/bart-large-mnli" or others if MoritzLaurer model has issues
        zsc_classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", # Good multilingual option
            # model="facebook/bart-large-mnli", # Another popular option
            device=target_device
        )
        # Process in batches if memory is a concern (adjust batch_size as needed)
        zsc_results = zsc_classifier(comment_list_cleaned, candidate_labels=CATEGORIES, multi_label=False, batch_size=8)
        print("Zero-shot classification complete.")
    except Exception as e:
        print(f"ERROR: Zero-shot classification failed. Details: {e}")
        # Create placeholder results if ZSC fails
        zsc_results = [{"labels": ["Error"], "scores": [0.0]}] * len(comment_list_cleaned)

    # --- Sentiment Analysis (Positive/Negative Labels) ---
    print(f"\nRunning sentiment analysis (Positive/Negative) on {len(comment_list_cleaned)} comments...")
    try:
        print(f"Attempting to run Sentiment Analysis on device: {'GPU 0' if target_device == 0 else 'CPU'}")
        sentiment_classifier = pipeline(
            "sentiment-analysis",
            # Using a standard English sentiment model as comments should be translated
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            device=target_device
        )
        # Process in batches if memory is a concern
        sentiment_results = sentiment_classifier(comment_list_cleaned, batch_size=8)
        print("Sentiment analysis complete.")
    except Exception as e:
        print(f"ERROR: Sentiment analysis failed. Details: {e}")
        # Create placeholder results if Sentiment fails
        sentiment_results = [{"label": "Error", "score": 0.0}] * len(comment_list_cleaned)

else:
    print("\nSkipping ML Analysis as no cleaned comments are available.")

# =========================================
# 6. Process Results & Generate Visualization
# =========================================
analysis_df = pd.DataFrame() # Initialize

# Ensure all results are available and match the length of the filtered DataFrame
if not comments_df_filtered.empty and zsc_results and sentiment_results and \
   len(zsc_results) == len(comments_df_filtered) and \
   len(sentiment_results) == len(comments_df_filtered):

    print("\nProcessing analysis results...")
    try:
        # Create a DataFrame from the analysis results
        analysis_results_df = pd.DataFrame({
            "top_category": [res["labels"][0] if res and "labels" in res and res["labels"] else "N/A" for res in zsc_results],
            "category_score": [res["scores"][0] if res and "scores" in res and res["scores"] else 0.0 for res in zsc_results],
            "sentiment_label": [res["label"].upper() if res and "label" in res else "Error" for res in sentiment_results],
            "sentiment_score": [res["score"] if res and "score" in res else 0.0 for res in sentiment_results],
            "cleaned_comment": comment_list_cleaned # Add cleaned comments used for analysis
        })

        # Reset index of both DataFrames to ensure proper alignment before joining
        comments_df_filtered = comments_df_filtered.reset_index(drop=True)
        analysis_results_df = analysis_results_df.reset_index(drop=True)

        # Combine analysis results with the filtered original comment data
        analysis_df = pd.concat([comments_df_filtered, analysis_results_df], axis=1)

        print(f"Created final analysis DataFrame with {len(analysis_df)} rows.")

        # --- Generate Bar Chart of Negative Comments per Category ---
        print("Generating bar chart for negative comments per category...")
        # Filter for negative comments based on the PIPELINE's output
        negative_comments_df = analysis_df[analysis_df['sentiment_label'] == 'NEGATIVE'].copy()

        if not negative_comments_df.empty:
            # Count occurrences of each category within negative comments
            negative_counts = negative_comments_df['top_category'].value_counts()

            # Ensure all defined categories are included, even if count is 0
            negative_counts = negative_counts.reindex(CATEGORIES, fill_value=0)
            # Filter out categories with 0 counts for cleaner chart, unless you want to show all
            negative_counts = negative_counts[negative_counts > 0]

            if not negative_counts.empty:
                 plt.figure(figsize=(12, 8)) # Adjust size as needed
                 bars = plt.bar(negative_counts.index, negative_counts.values, color='salmon')
                 plt.xlabel("Category")
                 plt.ylabel("Number of Negative Comments")
                 plt.title(f"Negative Comments per Category for '{PRODUCT_TO_SEARCH}' ({len(negative_comments_df)} Neg Comments Found)")
                 plt.xticks(rotation=45, ha='right') # Rotate labels for readability
                 plt.tight_layout() # Adjust layout

                 # Add counts on top of bars
                 for bar in bars:
                     yval = bar.get_height()
                     if yval > 0:
                         plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center')

                 plt.savefig(OUTPUT_CHART_FILE)
                 print(f"Bar chart saved to: {OUTPUT_CHART_FILE}")
                 # plt.show() # Optionally display inline
                 plt.close() # Close plot to free memory
            else:
                 print("No negative comments found for the specified categories after filtering.")

        else:
            print("No comments classified as NEGATIVE by the sentiment pipeline.")

    except Exception as e:
        print(f"ERROR: Failed to process results or generate/save the bar chart. Details: {e}")

elif comments_df_filtered.empty:
     print("\nSkipping result processing and chart generation as no comments survived preparation.")
else:
    print(f"\nWarning: Mismatch in result lengths or missing analysis results. Cannot process results or generate chart.")
    print(f"  Filtered DF rows: {len(comments_df_filtered)}")
    print(f"  ZSC results count: {len(zsc_results) if zsc_results else 'None'}")
    print(f"  Sentiment results count: {len(sentiment_results) if sentiment_results else 'None'}")


# =========================================
# 7. Save Detailed Analysis Results
# =========================================

if not analysis_df.empty:
    print(f"\nSaving detailed analysis results for {len(analysis_df)} comments to '{OUTPUT_CSV_FILE}'...")
    try:
        # Save to the current working directory (set earlier)
        analysis_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
        print(f"Saved detailed analysis CSV to: {os.path.join(os.getcwd(), OUTPUT_CSV_FILE)}")
    except Exception as e:
        print(f"ERROR: Failed to save detailed analysis CSV. Details: {e}")
else:
    print("\nSkipping saving detailed analysis CSV as the DataFrame is empty.")

# =========================================
# 8. Conclusion
# =========================================
print("\nScript finished.")