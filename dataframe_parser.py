"""
This module contains model files for analysis, primarily using GPT-4 for the model,
and CrewAI for agentic tasks.
"""

from crewai import Task, Agent, Crew


def dataframe_parser(df):
    """
    Parse a dataframe to extract characters mentioned in the 'content' column.

    This function sets up an agent using CrewAI to parse the dataframe. The agent
    reads each cell in the 'content' column of the dataframe to identify and
    extract character names.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data to be parsed. The dataframe
                           is expected to have a 'content' column with text content
                           generated from an SRT file.

    Returns:
    dict: A dictionary containing the results of the parsing task.
    """

    # Setting up the agent for parsing the dataframe
    parsing_agent = Agent(
        role="Dataframe parser",
        goal=("be the best parser and parse the dataframe with the highest accuracy. "
              "Your goal is to parse the dataframe and find out all the characters from the dataframe. "
              "Information about the characters is found in the content column."),
        backstory=("You are an agent working for the user. "
                   "Your goal is to provide the user with the parsed responses. "
                   "The parsing involves reading each and every cell of the content column. "
                   "After reading all the cells, find out the characters involved. "
                   "The dataframe was generated from an SRT file, so keep that in mind."),
        allow_delegation=False,
        verbose=True
    )

    # Defining the task for the agent
    parser_task = Task(
        description=("For the dataset {df}, the task at hand is to parse it and get accurate responses for the characters involved. "
                     "The work involves extreme precision, and the agent is not supposed to make any mistakes. "
                     "The agent is supposed to read each cell in the content column, understand it, and then come up with the characters involved."),
        expected_output=("A list of names corresponding to the characters. "
                         "The output must be without any errors."),
        agent=parsing_agent
    )

    # Setting up the crew with the agent and task
    crew = Crew(
        agents=[parsing_agent],
        tasks=[parser_task],
        verbose=2,
        memory=True
    )

    # Inputs for the crew task
    inputs = {
        "df": df
    }

    # Kicking off the task and returning the result
    result = crew.kickoff(inputs=inputs)
    return result
