import pandas as pd
import re

def parse_srt(srt_content):
    """
    Parses the content of an SRT file and extracts subtitle data.

    Parameters:
    srt_content (str): The content of the SRT file as a single string.

    Returns:
    list: A list of lists where each sublist contains the index, start time, end time, and text of a subtitle entry.
    """
    pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)\n\n', re.DOTALL)
    matches = pattern.findall(srt_content)

    data = []
    for match in matches:
        index = int(match[0])
        start_time = match[1]
        end_time = match[2]
        text = match[3].replace('\n', ' ')
        data.append([index, start_time, end_time, text])

    return data

def reader(path):
    """
    Reads an SRT file from the given path and converts it into a pandas DataFrame.

    Parameters:
    path (str): The path to the SRT file.

    Returns:
    pandas.DataFrame: A DataFrame containing the parsed subtitle data with columns 'Index', 'Start Time', 'End Time', and 'Content'.
    """
    with open(path, 'r', encoding='utf-8') as file:
        srt_content = file.read()
        parsed_data = parse_srt(srt_content)
        df = pd.DataFrame(parsed_data, columns=['Index', 'Start Time', 'End Time', 'Content'])
        return df


