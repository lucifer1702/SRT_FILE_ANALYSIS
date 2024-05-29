"""
Contains all the modules for the task at hand.
"""

from crewai import Agent, Task, Crew
from reader import reader
from dataframe_parser import dataframe_parser


def talking_speed_and_percentage(df, metadata):
    """
    Calculate the average talking speed and percentage of all characters in the SRT file.

    This function uses CrewAI to set up an agent that parses the dataframe to calculate the 
    talking speed and percentage of each character. The dataframe contains columns for start time, 
    end time, and content, and the metadata provides the names of the characters.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the SRT data with columns 'start time', 'end time', and 'content'.
    metadata (dict): A dictionary containing metadata about the characters, such as their names.

    Returns:
    dict: A dictionary with character names as keys and a list as values, where the first element is the talking speed 
          and the second element is the talking percentage.
    """
    # Setting up the agent for calculating talking speed and percentage
    talking_agent = Agent(
        role="calculating talking speed and percentage",
        goal=("your goal is to calculate the talking speed and talking percentage of all the characters in the SRT file. "
              "To do this, you shall utilize the metadata on characters: {metadata} and the dataframe: {df}. "
              "The dataframe contains 3 columns: the start time column, end time column, and the content column. "
              "Given the metadata on character names, your goal is to find out which sentences are spoken by the characters in the given metadata, "
              "then compute the average talking speed and average talking percentage."),
        backstory=("You are a highly aware and computational agent. "
                   "Your task is to work for the user and calculate all the talking speeds and percentages of a character. "
                   "You will carry out this task with the best efficiency, "
                   "understanding the task at hand with extreme precision and computing the results."),
        allow_delegation=False,
        verbose=True
    )

    # Defining the task for the agent
    talking_speed_task = Task(
        description=("The task in hand is to compute the average talking speed and percentage of all the characters in a dataframe: {df}. "
                     "The dataframe has been prepared from an SRT file and it has the start time, end time, and content. "
                     "Start time: time.time, end time: time.time, and content: str. "
                     "Your task is to use the metadata: {metadata} given to you, as it contains all the names of the characters, and then parse the dataframe. "
                     "Once you parse the dataframe, analyze each and every sentence the character says and find out the average talking speed, "
                     "and use the entire dataframe to compute the talking percentage of each character."),
        expected_output=("The expected output should be accurate and precise. "
                         "The output should be in Python dictionary format. "
                         "The keys should be the names of the characters, "
                         "and the values should be a list with the first element being the talking speed and the second being the talking percentage."),
        agent=talking_agent
    )

    # Setting up the crew with the agent and task
    crew = Crew(
        agents=[talking_agent],
        tasks=[talking_speed_task],
        verbose=2,
        memory=True
    )

    # Inputs for the crew task
    input = {
        "df": df,
        "metadata": metadata
    }

    # Kicking off the task and returning the result
    result = crew.kickoff(inputs=input)
    return result


def sentiment_analyser(df, metadata):
    """
    Compute the average sentiment of each character given a dataframe and metadata.

    This function uses CrewAI to set up an agent that parses the dataframe to calculate the average sentiment
    of each character. The dataframe contains columns for start time, end time, and content, and the metadata
    provides the names of the characters.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the SRT data with columns 'start time', 'end time', and 'content'.
    metadata (dict): A dictionary containing metadata about the characters, such as their names.

    Returns:
    dict: A dictionary with character names as keys and their average sentiment as values. The sentiment values can be 'positive', 'negative', or 'neutral'.
    """
    # Setting up the agent for computing average sentiment
    sentiment_agent = Agent(
        role="finding the average sentiment",
        goal=("Your goal is to compute the average sentiment of a character given a dataframe and the metadata. "
              "To do this, you shall utilize the metadata on characters: {metadata} and the dataframe: {df}. "
              "The dataframe contains 3 columns: the start time column, end time column, and the content column. "
              "Given the metadata on character names, your goal is to find out which sentences are spoken by the characters in the given metadata, "
              "then compute the average sentiment of each character by analyzing each record the character spoke. "
              "The sentiments can have 3 values: positive, negative, and neutral."),
        backstory=("You are a highly aware and computational agent. "
                   "Your task is to work for the user and compute the average sentiment for a given character. "
                   "You will carry out this task with the best efficiency, "
                   "understanding the task at hand with extreme precision and computing the results."),
        allow_delegation=False,
        verbose=True
    )

    # Defining the task for the agent
    sentiment_analyser_task = Task(
        description=("The task in hand is to compute the average sentiment of all the characters in a dataframe: {df}. "
                     "The dataframe has been prepared from an SRT file and it has the start time, end time, and content. "
                     "Start time: time.time, end time: time.time, and content: str. "
                     "Your task is to use the metadata: {metadata} given to you, as it contains all the names of the characters, and then parse the dataframe. "
                     "Once you parse the dataframe, analyze each and every sentence the character says and find out the average sentiment of the character throughout the entire dataframe. "
                     "The sentiments can take one of the 3 values: positive, negative, and neutral."),
        expected_output=("The expected output should be accurate and precise. "
                         "The output should be in Python dictionary format. "
                         "The keys should be the names of the characters, "
                         "and the values should be the sentiment of the character across the entire dataframe: type: str."),
        agent=sentiment_agent
    )

    # Setting up the crew with the agent and task
    crew = Crew(
        agents=[sentiment_agent],
        tasks=[sentiment_analyser_task],
        verbose=2,
        memory=True
    )

    # Inputs for the crew task
    input = {
        "df": df,
        "metadata": metadata
    }

    # Kicking off the task and returning the result
    result = crew.kickoff(inputs=input)
    return result
