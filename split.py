import pandas as pd
import re
import os

# Input CSV file path
INPUT_CSV_FILE = "reddit_analysis_Apple_Vision_Pro_posts_scraped.csv"  # Replace with your actual file name

# Output directory for the split CSV files
OUTPUT_DIR = "split_comments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of comments per output file
COMMENTS_PER_FILE = 10000


def process_comments(comment_string):
    """
    Treats the input string as a single comment and performs minimal cleaning.
    """
    comment_string = str(comment_string).strip()  # Ensure it's a string and remove leading/trailing whitespace
    if comment_string.lower() == '[removed]':
        return []  # Skip removed comments
    else:
        return [comment_string]  # Return as a list containing the single comment


# Load the CSV file
df = pd.read_csv(INPUT_CSV_FILE)

# Apply the processing function to the comments column
df['processed_comments'] = df['content'].astype(str).apply(process_comments)

# Flatten the list of lists into a single list of comments
all_comments = [comment for sublist in df['processed_comments'] for comment in sublist]

# Split the comments into chunks
num_chunks = (len(all_comments) + COMMENTS_PER_FILE - 1) // COMMENTS_PER_FILE
print(f"Total comments: {len(all_comments)}")
print(f"Number of chunks: {num_chunks}")

for i in range(num_chunks):
    start_index = i * COMMENTS_PER_FILE
    end_index = min((i + 1) * COMMENTS_PER_FILE, len(all_comments))
    chunk_comments = all_comments[start_index:end_index]

    # Create a DataFrame for the chunk
    chunk_df = pd.DataFrame({'comment': chunk_comments})

    # Define the output file name
    output_file = os.path.join(OUTPUT_DIR, f"reddit_post_chunk_{i + 1}.csv")

    # Export the DataFrame to CSV
    chunk_df.to_csv(output_file, index=False)

    print(f"Chunk {i + 1} exported to: {output_file}")

print("Splitting complete!")