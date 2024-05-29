"""
main.py file is responsible for the main execution of the program.
It is the entry point to the program.
"""

from model import talking_speed_and_percentage, sentiment_analyser
from reader import reader
from dataframe_parser import dataframe_parser
import os
import dotenv

# Load environment variables from a .env file
dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_MODEL_NAME'] = "gpt-4"

# Define the path to the input file
path = "path to file"
df = reader(path)
metadata = dataframe_parser(df)

def main():
    """
    The main function that orchestrates the execution of the program.

    This function performs the following tasks:
    1. Calculates the talking speed and percentage of characters in the provided dataframe.
    2. Analyzes the sentiment of the characters in the provided dataframe.
    3. Prints the results of both analyses.
    """
    result_speed_and_percentage = talking_speed_and_percentage(df=df, metadata=metadata)
    result_sentiment = sentiment_analyser(df=df, metadata=metadata)
    print(result_speed_and_percentage)
    print(result_sentiment)

if __name__ == "__main__":
    main()
