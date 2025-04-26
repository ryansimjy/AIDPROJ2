# Import necessary libraries
import time
import logging
import json
import os # For creating directories and handling paths
from urllib.parse import urlparse # For getting parts of the URL
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException, ElementClickInterceptedException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Configure logging (Set to INFO for production, DEBUG for details)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
webdriver_path = None # <-- SET THIS PATH IF chromedriver IS NOT IN YOUR SYSTEM PATH
OUTPUT_DIR = "vr_headset_data" # Directory to save JSON files
TARGET_URL = "https://vr-compare.com/headset/microsofthololens2" # The specific URL to scrape

# --- WebDriver Setup Function ---
def setup_driver(driver_path=None):
    """Sets up the Selenium WebDriver for local execution."""
    chrome_options = Options()
    # chrome_options.add_argument("--headless") # Uncomment for headless execution
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")

    # Suppress excessive Selenium logging below WARNING level
    selenium_logger = logging.getLogger('selenium.webdriver.remote.remote_connection')
    selenium_logger.setLevel(logging.WARNING)
    urllib3_logger = logging.getLogger('urllib3.connectionpool')
    urllib3_logger.setLevel(logging.WARNING)

    service = None
    if driver_path:
        try:
            service = Service(executable_path=driver_path)
            logging.info(f"Using WebDriver specified at: {driver_path}")
        except Exception as e:
            logging.error(f"Error initializing Service with path {driver_path}: {e}")
            return None
    try:
        if service:
             driver = webdriver.Chrome(service=service, options=chrome_options)
        else:
             logging.info("WebDriver path not specified, attempting to find chromedriver in PATH...")
             driver = webdriver.Chrome(options=chrome_options)
        logging.info("WebDriver initialized successfully.")
        return driver
    except WebDriverException as e:
        logging.error(f"Failed to initialize WebDriver: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during WebDriver setup: {e}")
        return None

# --- Cookie Consent Handler (JS Click) ---
def handle_cookie_consent(driver, timeout=5):
    """Checks for and clicks a common cookie consent button using JavaScript."""
    cookie_button_selector = "div.cookieBanner button"
    try:
        logging.info(f"Checking for cookie consent button with selector: {cookie_button_selector}")
        accept_button = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, cookie_button_selector))
        )
        logging.info("Cookie consent button found. Attempting JavaScript click.")
        driver.execute_script("arguments[0].click();", accept_button)
        logging.info("JavaScript click executed for cookie button.")
        time.sleep(1) # Short pause after click
    except TimeoutException:
        logging.info("Cookie consent button not found within timeout.")
    except NoSuchElementException:
        logging.info("Cookie consent button element does not exist.")
    except Exception as e:
        if isinstance(e, ElementClickInterceptedException):
             logging.error(f"JavaScript click failed or element still intercepted: {e}")
        else:
             logging.error(f"An error occurred while handling cookie consent: {e}")

# --- Scraping Function for Headset Details ---
def scrape_headset_details(url, driver):
    """
    Scrapes specification table, description, and FAQ from a single headset page.
    """
    headset_data = {
        "url": url,
        "specs": {},
        "description": None,
        "faq": []
    }
    try:
        logging.info(f"Attempting to fetch headset detail URL: {url}")
        driver.get(url)
        time.sleep(2) # Initial pause for page load

        # Handle potential cookie consent banner
        handle_cookie_consent(driver)
        time.sleep(1) # Pause after potential cookie handling

        # --- 1. Extract Specification Table ---
        logging.info("Attempting to extract specification table...")
        specs = {}
        try:
            spec_table_selector = "table.comparisonTable"
            # Wait longer for the table to ensure it's fully rendered
            table = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, spec_table_selector))
            )
            logging.info("Spec table element located.")
            row_selector = "tbody tr"
            # Wait for rows within the located table
            rows = WebDriverWait(table, 15).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, row_selector))
            )
            logging.info(f"Found {len(rows)} rows in the spec table body.")

            current_section = "General" # Default section name
            for i, row in enumerate(rows):
                logging.debug(f"Processing row {i}...")
                try:
                    # Check if it's a section heading row
                    row_classes = row.get_attribute('class') or ''
                    if 'sectionHeading' in row_classes:
                        # Find the heading text within the row
                        heading_text_element = row.find_element(By.CSS_SELECTOR, "td")
                        current_section = heading_text_element.text.strip()
                        if not current_section: current_section = "Other" # Fallback section name
                        logging.debug(f" Row {i} IS a section heading: {current_section}")
                        continue # Move to the next row

                    # If not a heading, try to extract spec name and value
                    spec_name_cells = row.find_elements(By.CSS_SELECTOR, "td.specHeadingCell")
                    spec_value_cells = row.find_elements(By.CSS_SELECTOR, "td.bodyCell")

                    if len(spec_name_cells) > 0 and len(spec_value_cells) > 0:
                        spec_name = spec_name_cells[0].text.strip()
                        # Get the text content of the value cell
                        spec_value = spec_value_cells[0].text.strip()

                        if spec_name: # Ensure spec name is not empty
                            logging.debug(f"  Row {i}: Section '{current_section}', Spec: '{spec_name}' = '{spec_value}'")
                            # Initialize section dictionary if it doesn't exist
                            if current_section not in specs:
                                specs[current_section] = {}
                            # Add the spec to the current section
                            specs[current_section][spec_name] = spec_value
                        else:
                            logging.debug(f"  Row {i}: Found spec cells but spec name was empty.")
                    else:
                        # Check if it's an ad row to avoid logging warnings for ads
                        ad_check = row.find_elements(By.CSS_SELECTOR, "div[id*='pw-']")
                        if not ad_check:
                             logging.debug(f"  Row {i}: Skipped - Did not find expected spec name/value cells. Row text: '{row.text[:100]}...'")
                        else:
                             logging.debug(f"  Row {i}: Skipped - Likely an ad row.")
                except Exception as row_err:
                    # Log error for the specific row but continue processing others
                    logging.error(f"Error processing row {i}: {row_err}. Row text: '{row.text[:100]}...'")
                    continue # Continue to the next row

            headset_data["specs"] = specs
            logging.info(f"Finished processing spec rows. Extracted {sum(len(v) for v in specs.values())} specs across {len(specs)} sections.")
        except TimeoutException:
            logging.warning("Timed out waiting for the specification table or its rows.")
        except Exception as e:
            logging.error(f"Error extracting specification table: {e}")

        # --- 2. Extract Description / About Section ---
        logging.info("Attempting to extract description...")
        try:
            description_selector = "div.headsetDescription"
            # Wait for the description element to be present
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, description_selector))
            )
            description_element = driver.find_element(By.CSS_SELECTOR, description_selector)
            # Get the text content of the description element
            description_text = description_element.text.strip()
            headset_data["description"] = description_text
            logging.info("Successfully extracted description.")
        except TimeoutException:
            logging.warning("Timed out waiting for the description section.")
        except NoSuchElementException:
             logging.warning("Description section element not found.")
        except Exception as e:
            logging.error(f"Error extracting description: {e}")

        # --- 3. Extract FAQ Section ---
        logging.info("Attempting to extract FAQ section...")
        faq_list = []
        try:
            faq_container_selector = "div.questionAndAnswer"
            # Wait for at least one FAQ element to be present
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, faq_container_selector))
            )
            faq_elements = driver.find_elements(By.CSS_SELECTOR, faq_container_selector)
            logging.info(f"Found {len(faq_elements)} potential FAQ entries.")
            for faq_element in faq_elements:
                try:
                    # Extract question and answer text from within each FAQ element
                    question = faq_element.find_element(By.CSS_SELECTOR, "h4.question").text.strip()
                    answer = faq_element.find_element(By.CSS_SELECTOR, "p.answer").text.strip()
                    # Add to list only if both question and answer are found
                    if question and answer:
                        faq_list.append({"question": question, "answer": answer})
                except NoSuchElementException:
                    logging.warning("Could not find question/answer structure within an FAQ element.")
                except Exception as faq_item_err:
                    logging.error(f"Error processing an FAQ item: {faq_item_err}")
            headset_data["faq"] = faq_list
            logging.info(f"Successfully extracted {len(faq_list)} FAQs.")
        except TimeoutException:
            logging.warning("Timed out waiting for the FAQ section (or no FAQs found).")
        except NoSuchElementException:
             logging.warning("FAQ section elements not found.")
        except Exception as e:
            logging.error(f"Error extracting FAQ: {e}")

        # Return the collected data
        return headset_data

    except WebDriverException as e:
        logging.error(f"Selenium WebDriver error during scraping of {url}: {e}")
        return None # Indicate failure
    except Exception as e:
        logging.error(f"An unexpected error occurred during scraping of {url}: {e}")
        return None # Indicate failure

# --- Main Execution Block ---
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        logging.info(f"Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

    print("\nSetting up Selenium WebDriver...")
    driver = setup_driver(webdriver_path)

    if driver:
        print("-" * 70)
        logging.info(f"Starting scraping for single URL: {TARGET_URL}")

        # Scrape details for the target URL
        details = scrape_headset_details(TARGET_URL, driver)

        if details:
            # Save details to JSON
            try:
                # Generate filename from the last part of the URL path
                parsed_url = urlparse(TARGET_URL)
                # Get the last part of the path, remove trailing slash if present
                slug = os.path.basename(parsed_url.path.rstrip('/'))
                if not slug: # Fallback if slug extraction fails
                     slug = "scraped_headset_data"
                filename = os.path.join(OUTPUT_DIR, f"{slug}.json")

                logging.info(f"Saving data for '{slug}' to {filename}")
                # Open the file for writing with UTF-8 encoding
                with open(filename, 'w', encoding='utf-8') as f:
                    # Dump the 'details' dictionary into the file as JSON
                    # indent=4 makes the JSON file human-readable
                    # ensure_ascii=False allows non-ASCII characters (like ™, ©)
                    json.dump(details, f, indent=4, ensure_ascii=False)
                logging.info(f"Successfully saved data to {filename}")

            except Exception as save_err:
                logging.error(f"Failed to save JSON for {TARGET_URL}: {save_err}")
        else:
            logging.error(f"Failed to scrape details for {TARGET_URL}. No JSON file saved.")

        # Clean up WebDriver
        logging.info("Closing WebDriver.")
        print("-" * 70)
        print("Closing WebDriver...")
        driver.quit()
        print("WebDriver closed.")

    else:
        print("\nFailed to initialize Selenium WebDriver. Scraping aborted.")

    print("\nScript finished.")

