from google import genai
import pandas as pd
import os
import sys

# Initialize the Gemini client (replace with your API key)
client = genai.Client(api_key="API KEY HERE")

# Define file paths and column names
COLUMN_NAME_WITH_COMMENTS = 'comment'
CATEGORIES = ["Price", "Material", "Comfort", "Headstrap", "Battery life", "Connectivity", "Weight"]

def analyze_comments(csv_file_path):
    """
    Analyzes comments from a CSV file using the Gemini API and exports the results to a new CSV.
    """
    print(f"Reading comments from: {csv_file_path}")
    df = pd.read_csv(csv_file_path)

    # Extract and preprocess the comments
    comments_to_process = df[COLUMN_NAME_WITH_COMMENTS].dropna().astype(str).tolist()
    comments_to_process = [comment for comment in comments_to_process if comment.strip()]

    # --- Prepare the Prompt for Single API Call ---
    categories_str = ", ".join(CATEGORIES)
    # Format comments with index numbers for easy matching with the response
    formatted_comments_for_prompt = "\n".join([f"{i+1}. \"{comment}\"" for i, comment in enumerate(comments_to_process)])

    # Construct the prompt asking the model to identify the most negative comment
    prompt = f"""Analyze each of the following comments, numbered 1 to {len(comments_to_process)}. For each comment, determine if it primarily discusses one of the following categories: {categories_str}.

    If a comment fits into ONE of these categories, identify that single category and determine if the sentiment expressed towards that topic is Positive or Negative.
    If a comment does not clearly fit into any of these specific categories, or discusses multiple categories. do NOT output.

    Return the analysis for each comment on a new line, starting with the comment itself [comment], followed by the category and sentiment.

    Comments to Analyze:
    {formatted_comments_for_prompt}

    Desired Output Format (one line per comment):
    [Comment]. Category: [Category Name], Sentiment: [Positive/Negative]
    OR

    Example Output:
    [Comment] Category: Price, Sentiment: Negative
    [Comment] Category: Comfort, Sentiment: Positive
    ...
    {len(comments_to_process)}. Category: Battery life, Sentiment: Negative

    Provide ONLY the results in the specified format, without any introductory text, concluding remarks, or other explanations."""

    try:
        # Get the response from the Gemini model
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17", contents=f"{prompt}"
        )

        # Process the response and store it in a list of dictionaries
        results = []
        for line in response.text.splitlines():
            if line.strip():  # Ignore empty lines
                try:
                    # Split the line into comment, category, and sentiment
                    parts = line.split(" Category: ")
                    comment = parts[0].strip()
                    category_sentiment = parts[1].split(", Sentiment: ")
                    category = category_sentiment[0].strip()
                    sentiment = category_sentiment[1].strip()

                    # Append the data to the results list
                    results.append({
                        "Comment": comment,
                        "Category": category,
                        "Sentiment": sentiment
                    })
                except IndexError:
                    print(f"Skipping line due to unexpected format: {line}")  # Handle unexpected formats

        # Create a Pandas DataFrame from the results
        df_results = pd.DataFrame(results)

        # Define the output CSV file path
        output_csv_file = csv_file_path.replace(".csv", "_analyzed.csv")  # Create a new name

        # Export the DataFrame to a CSV file
        df_results.to_csv(output_csv_file, index=False)  # index=False prevents writing the DataFrame index to the CSV

        # Print a confirmation message
        print(f"Analysis results exported to: {output_csv_file}")

    except Exception as e:
        print(f"An error occurred during analysis: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    # Hardcode the file path
    csv_file_path = "split_comments/reddit_comments_chunk_1.csv"  # Replace with your actual file path
    analyze_comments(csv_file_path)

    print("Analysis complete!")