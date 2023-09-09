import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import gpsd 
import time
import pynmea2

# Load the accident data
data = pd.read_csv("C:\\Users\\nares\\Desktop\\college\\Downloads\\unnati_phase1_data_revised.csv")

# Extract the speed and latitude and longitude columns
Speed = data["Speed"]
Lat = data["Lat"]
Long = data["Long"]

# Create a DBSCAN clustering model
model = DBSCAN(eps=0.01, min_samples=10)

# Fit the model to the data
model.fit(np.array([Lat, Long]).T)

# Get the cluster labels
labels = model.labels_

# Function to get the current speed of the mobile
def get_current_speed():
    # Get the current location of the mobile
    latitude, longitude = get_current_location()

    # Calculate the distance between the current location and the previous location
    distance = calculate_distance(latitude, longitude, previous_latitude, previous_longitude)

    # Calculate the time difference between the current time and the previous time
    time_difference = time.time() - previous_time

    # Calculate the current speed
    speed = distance / time_difference

    return speed

# Function to get the current location of the mobile using pynmea2
def get_current_location():
    # Use the pynmea2 module to get the current location of the mobile
    sentence = pynmea2.NMEASentence()
    sentence.parse(gpsd.get_current())

    latitude = sentence.latitude
    longitude = sentence.longitude

    return latitude, longitude

# Function to calculate the distance between two points in radians
def calculate_distance(latitude1, longitude1, latitude2, longitude2):
    # Convert latitude and longitude from degrees to radians
    latitude1 = np.radians(latitude1)
    longitude1 = np.radians(longitude1)
    latitude2 = np.radians(latitude2)
    longitude2 = np.radians(longitude2)

    # Use the Haversine formula to calculate the distance between two points
    distance = 6371.01 * np.arccos(
        np.sin(latitude1) * np.sin(latitude2) + np.cos(latitude1) * np.cos(latitude2) * np.cos(longitude1 - longitude2)
    )

    return distance

# Function to check if the current area is an accident-prone area
def is_accident_prone_area(latitude, longitude):
    # Add your logic to check if the given latitude and longitude are in an accident-prone area
    # You can use the DBSCAN clustering labels or any other method for this check
    # For now, let's assume that an area is accident-prone if label is 1
    return labels[i] == 1


# Function to send a notification
def send_notification(latitude, longitude, speed):
    # Replace this with your actual notification logic
    print(f"Notification: Accident-prone area detected at Lat: {latitude}, Long: {longitude}, Speed: {speed}")

    # Get the emergency contacts from the user
    emergency_contacts = []
    print("Please enter your emergency contacts, separated by commas:")
    contacts_str = input()
    for contact in contacts_str.split(","):
        emergency_contacts.append(contact)

    # Send an emergency message to the contacts
    for contact in emergency_contacts:
        print(f"Sending emergency message to {contact}")

# Main function
if __name__ == "__main__":
    # Initialize previous location and time
    previous_latitude, previous_longitude = 0.0, 0.0
    previous_time = time.time()

    # Get the current location of the mobile
    latitude, longitude = get_current_location()

    # Check if the current area is an accident-
