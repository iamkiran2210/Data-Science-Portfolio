import pandas as pd
import streamlit as st

@st.cache_data
def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the groundwater data from a CSV file into a pandas DataFrame.
    Uses Streamlit's caching to prevent reloading the data on each run.
    """
    try:
        df = pd.read_csv(filepath)
        # Convert column names to lowercase for consistency
        df.columns = df.columns.str.lower()
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at {filepath}")
        return pd.DataFrame()

def query_by_district(df: pd.DataFrame, district_name: str) -> pd.Series:
    """
    Queries the DataFrame for a specific district (case-insensitive).

    Args:
        df (pd.DataFrame): The DataFrame to query.
        district_name (str): The name of the district to find.

    Returns:
        pd.Series: The data for the found district, or None if not found.
    """
    if df.empty:
        return None

    # Perform a case-insensitive search
    result = df[df['district'].str.lower() == district_name.lower()]

    if not result.empty:
        # Return the first match as a Series
        return result.iloc[0]

    return None
