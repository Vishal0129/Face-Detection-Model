import cx_Oracle
from datetime import datetime, timedelta
import time
import json

class Database:
    def __init__(self, username, password, host, port, service_name):
        dsn = cx_Oracle.makedsn(host, port, service_name=service_name)
        self.connection = cx_Oracle.connect(username, password, dsn)

    def get_unique_encounters(self, time_period_minutes):
        query = f"""
            SELECT e1.NAME, e1.CONFIDENCE, e1.TIMESTAMP, e1.IMAGE, e1.CAMERASOCKETURL, e1.LOCATION
            FROM encounters e1
            JOIN (
                SELECT NAME, MAX(TIMESTAMP) AS max_timestamp
                FROM encounters
                WHERE TO_DATE(TIMESTAMP, 'DD-MON-YYYY HH24:MI:SS') >= :start_time
                GROUP BY NAME
            ) e2 ON e1.NAME = e2.NAME AND e1.TIMESTAMP = e2.max_timestamp
        """
        start_time = datetime.now() - timedelta(minutes=time_period_minutes)
        with self.connection.cursor() as cursor:
            cursor.execute(query, start_time=start_time)
            results = cursor.fetchall()
            
        return results








if __name__ == "__main__":
    db = Database(username="aimlb4",
                  password="vishalsai",
                  host="localhost",
                  port="1521",
                  service_name="xe")

    time_period_minutes = 5  # Adjust the time period as needed

    while True:
        encounters = db.get_unique_encounters(time_period_minutes)
        encounter_list = []
        for encounter in encounters:
            # Assuming you have a list of column names to work with
            column_names = ["NAME", "CONFIDENCE", "TIMESTAMP", "IMAGE", "CAMERASOCKETURL", "LOCATION"]
            encounter_details = {column: value for column, value in zip(column_names, encounter)}

            # Append the encounter details to the list
            encounter_list.append(encounter_details)

        # Convert the list of encounters to JSON format
        encounters_json = json.dumps(encounter_list)

        # Print the JSON results
        print(encounters_json)

        time.sleep(60)
