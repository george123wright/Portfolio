"""
Scrapes posts and comments from r/wallstreetbets, extracts ticker mentions, analyses sentiment using NLTK’s VADER, 
and saves aggregated results with conditional formatting in Excel.
"""

import re
import time
import logging
import requests
import datetime as dt
import pandas as pd
from collections import Counter
from typing import List, Dict, Any

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from openpyxl.formatting.rule import CellIsRule
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
import config


nltk.download('vader_lexicon', quiet=True)

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()

console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

console_handler.setFormatter(console_formatter)

logger.addHandler(console_handler)

STOPWORDS = {
    "the", "and", "to", "a", "in", "of", "for", "is", "on", "that",
    "with", "as", "at", "this", "it", "by", "from", "are", "an", "i",
    "you", "but", "if", "or", "be", "was", "so", "we", "they", "not",
    "have", "has", "im", "can", "there", "what", "will", "all", "just",
    "https", "s", "trump", "ukraine", "minerals", "there", "why", "wondering",
    "apparently", "everyone", "there", "says", "their", "about",
    "reddit", "r", "wallstreetbets", "amp", "http", "www", "com",
    "figured", "last", "coffee", "people", "college", "university", 
    "student", "obvious", "me", "bpunt", "slurping", "see", "guys", 
    "should", "do", "people", "per", "usually"
}

NOTTICKERS = {
    "CEO", "CFO", "CTO", "COO", "WSB", "DD", "YOLO", "TOS", "AOC", "GDP",
    "OTM", "GAIN", "IS", "UK", "THE", "US", "NEVER", "OK", "FDA", "AM", "PM",
    "RH", "EV", "IPO", "ATH", "TOS", "TD", "EDIT", "TLDR", "ROPE", "STAY", "SAFE",
    "AUTO", "BOT", "AI", "IT", "ELON", "MUSK", "SEC", "TICK", "TOCK", "USD", "CPU",
    "IS", "DOE", "OP", "NY", "DOJ", "IRA", "NOT", "ZERO", "II", "III", "IV",
    "NFA", "IN", "BUY", "BUT", "JOBS", "THEIR", "WAS", "GOOD", "EU", "DVD", "EPS",
    "IM", "RIF"
}


def get_wsb_posts(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch posts from r/wallstreetbets via Reddit’s public API. If rate-limited or any request
    exception occurs, logs a warning and returns an empty list.

    Args:
        limit (int): Number of top posts to fetch (default is 100).

    Returns:
        A list of post dictionaries (empty list if an error occurs).
    """

    url = f"https://www.reddit.com/r/wallstreetbets/.json?limit={limit}"

    headers = {'User-Agent': 'Mozilla/5.0 (compatible; TickerScraper/1.0)'}

    try:

        response = requests.get(url, headers=headers, timeout=10)

        response.raise_for_status()

    except requests.RequestException as exc:

        logger.error("Error fetching posts from Reddit: %s", exc)

        return []

    data = response.json()

    return data.get("data", {}).get("children", [])


def get_comments_for_post(post_id: str) -> List[str]:
    """
    Fetch all top-level and nested comments from a given post ID in r/wallstreetbets.
    Employs a small retry loop with exponential backoff if rate-limited.

    Args:
        post_id (str): The Reddit post ID.

    Returns:
        A list of comment texts. Returns an empty list on failure or if no comments are found.
    """

    url = f"https://www.reddit.com/r/wallstreetbets/comments/{post_id}/.json"

    headers = {'User-Agent': 'Mozilla/5.0 (compatible; TickerScraper/1.0)'}

    max_retries = 2

    delay = 5

    for attempt in range(max_retries):

        try:

            response = requests.get(url, headers=headers, timeout=10)

            response.raise_for_status()

            break
       
        except requests.RequestException as exc:

            if "429" in str(exc):

                logger.warning(
                    "Rate limited fetching comments for post %s (attempt %d/%d). Sleeping %d seconds.",
                    post_id, attempt + 1, max_retries, delay
                )

                time.sleep(delay)

                delay *= 2

            else:

                logger.error("Error fetching comments for post %s: %s", post_id, exc)

                return []
    else:

        logger.error("Failed to fetch comments for post %s after %d retries.", post_id, max_retries)

        return []

    data = response.json()

    if len(data) < 2:

        return []

    comment_data = data[1].get("data", {}).get("children", [])

    comments = []

    def extract_comments(comments_list: List[Dict[str, Any]]) -> None:
        """
        Recursively traverse comment threads and collect comment bodies.
        """

        for item in comments_list:

            if item.get("kind") != "t1":

                continue

            comment_body = item.get("data", {}).get("body", "")

            if comment_body:

                comments.append(comment_body)

            replies = item.get("data", {}).get("replies")

            if replies and isinstance(replies, dict):

                next_children = replies.get("data", {}).get("children", [])

                extract_comments(next_children)

    extract_comments(comment_data)

    return comments


def extract_tickers(text: str) -> List[str]:
    """
    Extract potential stock ticker symbols from a given text, ignoring items from NOTTICKERS.

    Args:
        text (str): Input text to search for ticker candidates.

    Returns:
        A list of uppercase ticker symbols, e.g. ["AAPL", "GOOG"], not in NOTTICKERS.
    """

    pattern = r'\b[A-Z]{2,5}\b'

    tickers = re.findall(pattern, text)

    return [ticker for ticker in tickers if ticker not in NOTTICKERS]


def extract_words(text: str) -> List[str]:
    """
    Extract lowercased words from text, filtering out a predefined set of stopwords.

    Args:
        text (str): The text to tokenize.

    Returns:
        A list of words excluding stopwords.
    """

    words = re.findall(r'\b\w+\b', text.lower())

    return [word for word in words if word not in STOPWORDS]


def format_sheet_as_table(excel_file: str, sheet_name: str) -> None:
    """
    Formats an existing sheet in an Excel file as a table for improved readability.

    Args:
        excel_file (str): Path to the Excel file.
        sheet_name (str): Name of the sheet to format.
    """

    try:

        wb = load_workbook(excel_file)

        if sheet_name not in wb.sheetnames:

            logger.warning("Sheet '%s' not found in %s", sheet_name, excel_file)

            return

        ws = wb[sheet_name]

        max_row = ws.max_row

        max_col = ws.max_column

        last_col_letter = get_column_letter(max_col)

        table_range = f"A1:{last_col_letter}{max_row}"

        table_name = sheet_name.replace(" ", "") + "Table"

        table = Table(displayName=table_name, ref=table_range)

        style = TableStyleInfo(
            name="TableStyleMedium9",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False
        )

        table.tableStyleInfo = style

        ws.add_table(table)

        wb.save(excel_file)

        logger.info("Sheet '%s' formatted as a table in %s.", sheet_name, excel_file)

    except Exception as exc:

        logger.error("Error formatting sheet '%s' as table: %s", sheet_name, exc)


def main() -> None:
    """
    Main function that:
      1. Scrapes the latest WallStreetBets posts.
      2. Extracts ticker mentions, comment texts, and performs sentiment analysis.
      3. Aggregates ticker mentions, average sentiment, top words, etc.
      4. Saves results to Excel and applies conditional formatting and table styling.
    """

    logger.info("Scraping WallStreetBets posts...")

    posts = get_wsb_posts(limit=100)

    if not posts:

        logger.warning("No posts fetched. Exiting.")

        return

    sia = SentimentIntensityAnalyzer()

    custom_words = {
        "buy": 4.0, "sell": -4.0, "bull": 4, "bullish": 4, "bear": -4,
        "bearish": -4, "moon": 4, "rocket": 4, "crash": -4, "yolo": 2,
        "tendies": 2, "stonks": 2, "omg": 1, "fomo": 1, "fml": -1,
        "up": 4, "down": -4, "long": 4, "short": -4, "overvalued": -3,
        "undervalued": 3, "pump": 4, "dump": -4, "bagholder": -2,
        "moonshot": 2, "highs": 2, "lows": -2, "underrated": 3,
        "overrated": -3, "pumping": 2, "dumping": -3, "recession": -2,
        "depression": -2, "crisis": -2, "soaring": 3, "soars": 2,
        "soared": 2, "plunges": -3, "plunged": -2, "plunge": -3,
        "surge": 4, "surges": 4, "surged": 2, "collapses": -4,
        "collapsed": -2, "collapse": -4, "rally": 2, "rallies": 2,
        "rallied": 2, "crashes": -4, "crashed": -2, "buy the dip": 4,
        "put": -4, "call": 4, "buying": 4, "selling": -4, "cheap": 4,
        "expensive": -4, "late": -4
    }

    sia.lexicon.update(custom_words)

    ticker_word_counter: Dict[str, Counter] = {}

    ticker_mentions: Dict[str, int] = {}

    ticker_sentiments: Dict[str, List[float]] = {}

    logger.info("Processing %d posts (including comments)...", len(posts))

    for i, post in enumerate(posts):


        logger.info("Processing Post %d/%d", i + 1, len(posts))

        post_data = post.get("data", {})

        post_id = post_data.get("id", "")

        title = post_data.get("title", "")

        selftext = post_data.get("selftext", "")

        combined_post_text = f"{title} {selftext}"

        post_tickers = extract_tickers(title)

        if post_tickers:

            sentiment_scores = sia.polarity_scores(combined_post_text)

            words_in_post = extract_words(combined_post_text)

            for tkr in post_tickers:

                ticker_mentions[tkr] = ticker_mentions.get(tkr, 0) + 1

                ticker_word_counter.setdefault(tkr, Counter()).update(words_in_post)

                ticker_sentiments.setdefault(tkr, []).append(sentiment_scores['compound'])

        comments = get_comments_for_post(post_id)

        for comment_text in comments:

            comment_tickers = extract_tickers(comment_text)

            if not comment_tickers:

                continue

            comment_sentiment = sia.polarity_scores(comment_text)['compound']

            comment_words = extract_words(comment_text)

            for tkr in comment_tickers:

                ticker_mentions[tkr] = ticker_mentions.get(tkr, 0) + 1

                ticker_word_counter.setdefault(tkr, Counter()).update(comment_words)

                ticker_sentiments.setdefault(tkr, []).append(comment_sentiment)

        time.sleep(1)

    ticker_avg_sentiment = {}

    ticker_sentiment_std = {}

    for tkr, sentiment_values in ticker_sentiments.items():

        avg_score = sum(sentiment_values) / len(sentiment_values)

        ticker_avg_sentiment[tkr] = avg_score

        if len(sentiment_values) > 1:

            variance = sum((val - avg_score) ** 2 for val in sentiment_values) / len(sentiment_values)

            ticker_sentiment_std[tkr] = variance ** 0.5

        else:

            ticker_sentiment_std[tkr] = 0.0

    ticker_top_words = {
        tkr: ", ".join([word for word, _ in ticker_word_counter[tkr].most_common(3)])
        for tkr in ticker_word_counter
    }

    sorted_tickers = sorted(ticker_mentions, key=ticker_mentions.get, reverse=True)

    findings = []

    for tkr in sorted_tickers:

        findings.append({
            "ticker": tkr,
            "mentions": ticker_mentions[tkr],
            "avg_sentiment": ticker_avg_sentiment.get(tkr, 0),
            "sentiment_std": ticker_sentiment_std.get(tkr, 0),
            "top_words": ticker_top_words.get(tkr, "")
        })

    findings_df = pd.DataFrame(findings)

    try:

        with pd.ExcelWriter(config.FORECAST_FILE, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:

            findings_df.to_excel(writer, sheet_name='Sentiment Findings', index=False)

            ws = writer.sheets['Sentiment Findings']

            try:

                col_idx = findings_df.columns.get_loc("avg_sentiment") + 1  

                col_letter = get_column_letter(col_idx)

            except Exception as exc:

                logger.error("Error finding the 'avg_sentiment' column index: %s", exc)

                col_letter = "C"  

            max_row = ws.max_row

            red_fill = PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid')

            green_fill = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')

            ws.conditional_formatting.add(
                f"{col_letter}2:{col_letter}{max_row}",
                CellIsRule(operator='lessThan', formula=['0'], fill=red_fill)
            )
         
            ws.conditional_formatting.add(
                f"{col_letter}2:{col_letter}{max_row}",
                CellIsRule(operator='greaterThan', formula=['0'], fill=green_fill)
            )

        logger.info("Sentiment findings successfully saved to Excel in '%s'.", config.FORECAST_FILE)

    except Exception as exc:
      
        logger.error("Failed to write sentiment findings to Excel: %s", exc)

    format_sheet_as_table(config.FORECAST_FILE, 'Sentiment Findings')


if __name__ == "__main__":
    main()
