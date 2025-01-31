from ratelimit import limits, sleep_and_retry
import argparse
import json
import MeCab
import requests
import logging
import sys

logging.basicConfig(filename="errors.log", level=logging.ERROR,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Define rate-limiting parameters
TEN_SECONDS = 10
MAX_CALLS = 10


@sleep_and_retry
@limits(calls=MAX_CALLS, period=TEN_SECONDS)
def fetch_word_from_jisho(word):
    """
    Fetches information about a single Japanese word from Jisho.org API.
    Enforces rate-limiting to avoid exceeding API limits.
    """
    try:
        url = f"https://jisho.org/api/v1/search/words?keyword={word}"
        response = requests.get(url, timeout=5)  # Timeout for network stability
        response.raise_for_status()  # Raises an HTTPError for 4xx/5xx responses

        data = response.json()
        if not data.get('data'):  # Avoid KeyError by using .get()
            return {"error": f"No data found for '{word}'."}

        word_data = data['data'][0]
        return {
            "word": word_data.get('slug', word),  # Fallback to input word if missing
            "reading": word_data['japanese'][0].get('reading', 'N/A') if 'japanese' in word_data else 'N/A',
            "meanings": [s.get('english_definitions', []) for s in word_data.get('senses', [])]
        }

    except requests.exceptions.Timeout:
        logging.error(f"Request timed out for '{word}'")
        return {"error": f"Request timed out for '{word}'."}

    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error for '{word}': {e}")
        return {"error": f"HTTP error: {e}"}

    except requests.exceptions.ConnectionError:
        logging.error(f"Connection error while fetching '{word}'")
        return {"error": "Network connection error."}

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch data for '{word}': {e}")
        return {"error": "Network error occurred."}

    except json.JSONDecodeError:
        logging.error(f"Failed to parse JSON for '{word}'")
        return {"error": "Invalid response format from Jisho API."}

    except KeyError as e:
        logging.error(f"Missing expected key {e} in response for '{word}'")
        return {"error": f"Missing data field: {e}"}

    except Exception as e:
        logging.error(f"Unexpected error for '{word}': {e}")
        return {"error": "Unexpected error occurred."}


def fetch_words_batch_from_jisho(input_words):
    """
    Fetches information for a list of Japanese words from Jisho.org API in batches with rate-limiting.
    :param input_words: list - A list of Japanese words.
    :return: dict - A dictionary mapping words to their data or error messages.
    """
    return {word: fetch_word_from_jisho(word) for word in set(input_words)}


def tokenize_text_with_pos(text):
    """
    Tokenizes a given Japanese text into individual words or morphemes using MeCab
    and includes detailed part-of-speech (POS) tagging, conjugation details, and sub-POS.
    """
    try:
        tagger = MeCab.Tagger()
        if not tagger:
            raise RuntimeError("Failed to initialize MeCab.")

    except RuntimeError as e:
        error_message = f"MeCab initialization error: {e}. Ensure MeCab is installed."
        logging.error(error_message)
        return {"error": error_message}

    except Exception as e:
        error_message = f"Unexpected error initializing MeCab: {e}"
        logging.error(error_message)
        return {"error": error_message}

    try:
        node = tagger.parseToNode(text)
        if node is None:
            raise ValueError("MeCab failed to parse the text.")

        tokens = []
        while node:
            surface = node.surface
            if surface:
                details = node.feature.split(',')
                tokens.append({
                    "surface": surface,
                    "pos": details[0] if len(details) > 0 else "Unknown",
                    "sub_pos": details[1] if len(details) > 1 else "Unknown",
                    "conjugation_type": details[4] if len(details) > 4 else "None",
                    "conjugation_form": details[5] if len(details) > 5 else "None",
                    "base_form": details[6] if len(details) > 6 else surface,
                    "reading": details[7] if len(details) > 7 else "N/A",
                    "pronunciation": details[8] if len(details) > 8 else "N/A"
                })
            node = node.next

        if not tokens:
            raise ValueError("No tokens were found.")

        return tokens

    except ValueError as e:
        error_message = f"MeCab processing error: {e}"
        logging.error(error_message)
        return {"error": error_message}

    except Exception as e:
        error_message = f"Unexpected error while tokenizing: {e}"
        logging.error(error_message)
        return {"error": error_message}


def export_results_to_json(tokens, word_data_map, file_path="results.json"):
    """
    Exports tokenized text and word meanings to a JSON file.
    """
    results = {
        "tokens": tokens,
        "word_data": word_data_map
    }

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

    print(f"Results exported to {file_path}")


def cli():
    """
    Command Line Interface for the tokenizer and batch fetch tool.
    Supports both direct text input and text file input.
    """
    parser = argparse.ArgumentParser(
        description="Japanese Text Tokenizer and Jisho.org Word Fetcher"
    )
    parser.add_argument(
        "text",
        type=str,
        help="The Japanese text to tokenize. Provide a string or a file path (prefix file input with '@')."
    )
    parser.add_argument(
        "-o", "--output",
        type=argparse.FileType("w", encoding="utf-8"),
        help="Path to save results as a JSON file (default: results.json).",
        default="results.json"
    )
    args = parser.parse_args()

    # Check if input is a file (prefix '@' indicates file input)
    if args.text.startswith("@"):
        file_path = args.text[1:]  # Remove '@' prefix
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read().strip()
        except IOError as e:
            print(f"Error reading file '{file_path}': {e}")
            sys.exit(1)
    else:
        text = args.text.strip()

    print("Tokenizing the input text...")
    tokens = tokenize_text_with_pos(text)
    if "error" in tokens:
        print(f"Error: {tokens['error']}")
        sys.exit(1)

    print("Tokenized Text:")
    for token in tokens:
        print(f"- {token['surface']} ({token['pos']})")

    print("\nFetching meanings from Jisho.org...")
    words = [token["surface"] for token in tokens]
    word_data_map = fetch_words_batch_from_jisho(words)

    for word, data in word_data_map.items():
        if "error" in data:
            print(f"Word: {word} - Error: {data['error']}")
        else:
            meanings = ", ".join(sum(data["meanings"], []))
            print(f"Word: {word} - Reading: {data['reading']} - Meanings: {meanings}")

    try:
        export_results_to_json(tokens, word_data_map, args.output.name)
    except IOError as e:
        print(f"Error writing to file: {e}")
        logging.error(f"File writing error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
