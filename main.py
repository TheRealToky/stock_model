import datetime as dt

# Get the current date and time
today = dt.datetime.today()

# Define the time difference (2 days ago)
two_days_ago = today - dt.timedelta(days=4)

# Print the results
print("Current date and time:", today)
print("Date and time two days ago:", two_days_ago)