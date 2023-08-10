# import os
from datetime import datetime
DATE_FORMAT = "%Y%m%d_%H%M%S_%f"
now = datetime.now().strftime(DATE_FORMAT)[:-3]
print(now)
timestamp = datetime.strptime(now, DATE_FORMAT)
print(timestamp)