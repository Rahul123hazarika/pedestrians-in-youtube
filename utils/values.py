import common
from custom_logger import CustomLogger
from logmod import logs
import warnings
import ast
import pandas as pd

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


class Values():
    def __init__(self) -> None:
        pass

    def find_values_with_video_id(self, df, key):
        """Extracts relevant data from a DataFrame based on a given key.

        Args:
            df (DataFrame): The DataFrame containing the data.
            key (str): The key to search for in the DataFrame.

        Returns:
            tuple: A tuple containing information related to the key, including:
                - Video ID
                - Start time
                - End time
                - Time of day
                - City
                - State
                - Latitude
                - Longitude
                - Country
                - GDP per capita
                - Population
                - Population of the country
                - Traffic mortality
                - Continent
                - Literacy rate
                - Average height
                - ISO-3 code for country
                - Fps of the video
                - Type of vehicle
        """
        id, start_, fps = key.rsplit("_", 2)  # Splitting the key into video ID and start time

        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            # Extracting data from the DataFrame row
            video_ids = [id.strip() for id in row["videos"].strip("[]").split(',')]
            start_times = ast.literal_eval(row["start_time"])
            end_times = ast.literal_eval(row["end_time"])
            time_of_day = ast.literal_eval(row["time_of_day"])
            city = row["city"]
            state = row['state'] if not pd.isna(row['state']) else "unknown"
            latitude = row["lat"]
            longitude = row["lon"]
            country = row["country"]
            gdp = row["gmp"]
            population = row["population_city"]
            population_country = row["population_country"]
            traffic_mortality = row["traffic_mortality"]
            continent = row["continent"]
            literacy_rate = row["literacy_rate"]
            avg_height = row["avg_height"]
            iso3 = row["iso3"]
            vehicle_type = ast.literal_eval(row["vehicle_type"])

            # Iterate through each video, start time, end time, and time of day
            for video, start, end, time_of_day_, vehicle_type, in zip(video_ids, start_times, end_times, time_of_day, vehicle_type):  # noqa: E501
                # Check if the current video matches the specified ID
                if video == id:
                    logger.debug(f"Finding values for {video} start={start}, end={end}")
                    counter = 0
                    # Iterate through each start time
                    for s in start:
                        # Check if the start time matches the specified start time
                        if int(start_) == s:
                            # Calculate gpd per capita to avoid division by zero
                            if int(population) > 0:
                                gpd_capita = int(gdp)/int(population)
                            else:
                                gpd_capita = 0
                            # Return relevant information once found
                            return (video,                      # 0
                                    s,                          # 1
                                    end[counter],               # 2
                                    time_of_day_[counter],      # 3
                                    city,                       # 4
                                    state,                      # 5
                                    latitude,                   # 6
                                    longitude,                  # 7
                                    country,                    # 8
                                    gpd_capita,                 # 9
                                    population,                 # 10
                                    population_country,         # 11
                                    traffic_mortality,          # 12
                                    continent,                  # 13
                                    literacy_rate,              # 14
                                    avg_height,                 # 15
                                    iso3,                       # 16
                                    int(fps),                   # 17
                                    vehicle_type)               # 18
                        counter += 1

    def get_value(self, df, column_name1, column_value1, column_name2, column_value2, target_column):
        """
        Retrieves a value from the target_column based on the condition
        that both column_name1 matches column_value1 and column_name2 matches column_value2.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the mapping file.
        column_name1 (str): The first column to search for the matching value.
        column_value1 (str): The value to search for in column_name1.
        column_name2 (str): The second column to search for the matching value.
        column_value2 (str): The value to search for in column_name2. If "unknown", the value is treated as NaN.
        target_column (str): The column from which to retrieve the corresponding value.

        Returns:
        Any: The value from target_column that corresponds to the matching values in both
             column_name1 and column_name2.
        """
        if column_name2 is None or column_value2 is None:
            result = df[df[column_name1] == column_value1][target_column]
            if not result.empty:
                return result.iloc[0]
            else:
                return None

        else:
            # Treat column_value2 as NaN if it is "unknown"
            if column_value2 == "unknown":
                column_value2 = float('nan')

            # Filter the DataFrame where both conditions are met
            if pd.isna(column_value2):
                result = df[(df[column_name1] == column_value1) & (df[column_name2].isna())][target_column]
            else:
                result = df[(df[column_name1] == column_value1) & (df[column_name2] == column_value2)][target_column]

            # Check if the result is not empty (i.e., if there is a match)
            if not result.empty:
                # Return the first matched value
                return result.values[0]
            else:
                # Return None if no matching value is found
                return None

    def calculate_total_seconds_for_city(self, df, city_name, state_name):
        """
        Calculates the total seconds of video for a specific city and state according to mapping file.
        """
        # Filter the DataFrame for the specific city and state
        if state_name.lower() == "unknown":
            row = df[(df['city'] == city_name) & (pd.isna(df['state']))]
        else:
            row = df[(df['city'] == city_name) & (df['state'] == state_name)]
        if row.empty:
            return 0  # or raise an error if city not found

        row = row.iloc[0]  # get the first matching row

        # Extracting data from the row
        start_times = ast.literal_eval(row["start_time"])
        end_times = ast.literal_eval(row["end_time"])

        total_seconds = 0

        # Iterate through each start time and end time
        for start, end in zip(start_times, end_times):
            for s, e in zip(start, end):
                total_seconds += (int(e) - int(s))

        return total_seconds

    def remove_columns_below_threshold(self, df, threshold):
        """
        Removes columns from the DataFrame if their total seconds (per city) is below a given threshold.
        Returns the modified DataFrame.
        """
        cols_to_remove = []

        # Assume all city/state rows, so iterate over columns that have start/end times
        for col in df.columns:
            # Check if this column is a 'start_time' or 'end_time' column
            if col.startswith("start_time") or col.startswith("end_time"):
                # Derive base name to match start/end pairs
                base = col.replace("start_time", "").replace("end_time", "")
                start_col = f"start_time{base}"
                end_col = f"end_time{base}"

                if start_col in df.columns and end_col in df.columns:
                    total_seconds = 0
                    for idx, row in df.iterrows():
                        start_times = ast.literal_eval(row[start_col])
                        end_times = ast.literal_eval(row[end_col])
                        for start, end in zip(start_times, end_times):
                            for s, e in zip(start, end):
                                total_seconds += (int(e) - int(s))
                    if total_seconds < threshold:
                        cols_to_remove += [start_col, end_col]

        # Remove duplicate column names just in case
        cols_to_remove = list(set(cols_to_remove))
        df_modified = df.drop(columns=cols_to_remove)
        return df_modified
