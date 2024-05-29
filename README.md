# SRT FILE ANALYSER

Subtitle Analysis using CrewAI and GPT-4

## Overview

This project aims to analyze subtitles from an SRT file using various AI techniques. The analysis includes calculating the talking speed and percentage of characters and performing sentiment analysis on the dialogues. The project utilizes CrewAI for agent-based tasks and GPT-4 for advanced text processing.

## Structure

The project consists of the following main components:

- `main.py`: The entry point for the program.
- `model.py`: Contains functions for performing various analyses on the subtitle data.
- `reader.py`: Reads and processes the subtitle file.
- `dataframe_parser.py`: Parses the dataframe to extract relevant information about the characters.
- `.env`: Contains environment variables such as API keys.

## Installation

1. **Clone the repository**:

   ```
   git clone https://github.com/yourusername/subtitle-analysis.git
   cd subtitle-analysis
   ```
2. **Create and activate a virtual environment** (optional but recommended):

   ```
   python3 -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```
3. **Install the required packages**:

   ```
   pip install -r requirements.txt
   ```
4. **Set up environment variables**:

   - Create a `.env` file in the root directory of your project and add the following lines:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```

## Usage

### Running the Program

To run the program, execute the `main.py` file

### Components

#### `main.py`

This is the main execution file and the entry point of the program. It performs the following tasks:

1. Reads the subtitle file and processes it into a dataframe.
2. Extracts metadata from the dataframe.
3. Calculates the talking speed and percentage of characters.
4. Analyzes the sentiment of the characters.
5. Prints the results.

#### `model.py`

Contains the core functions for analysis:

- `talking_speed_and_percentage(df, metadata)`: Calculates the talking speed and percentage of each character.
- `sentiment_analyser(df, metadata)`: Analyzes the sentiment of each character's dialogues.

#### `reader.py`

Defines the `reader(path)` function that reads the subtitle file from the given path and converts it into a dataframe.

#### `dataframe_parser.py`

Defines the `dataframe_parser(df)` function that parses the dataframe to extract character information.

## Example

Here is an example of how to set up and run the project:

1. Place your SRT file in the project directory.
2. Update the `path` variable in `main.py` to point to your SRT file.
3. Run the main script

## Output

The program will print two sets of results:

1. Talking speed and percentage of each character.
2. Sentiment analysis of each character's dialogues.

The results will be displayed in the console in dictionary format.
