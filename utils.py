
import pandas as pd

def load_all_data():
    gps = pd.read_csv("DATA/CFC GPS Data.csv", encoding="ISO-8859-1")
    gps['date'] = pd.to_datetime(gps['date'], dayfirst=True)

    recovery = pd.read_csv("DATA/CFC Recovery status Data.csv")
    recovery['sessionDate'] = pd.to_datetime(recovery['sessionDate'], dayfirst=True)

    physical = pd.read_csv("DATA/CFC Physical Capability Data_.csv")
    physical['testDate'] = pd.to_datetime(physical['testDate'], dayfirst=True)

    priority = pd.read_csv("DATA/CFC Individual Priority Areas.csv")

    return gps, recovery, physical, priority

def calculate_injury_risk(row, df):
    risk = 0
    if row['distance_over_24'] > df['distance_over_24'].mean() + df['distance_over_24'].std():
        risk += 0.3
    if row['accel_decel_over_2_5'] > df['accel_decel_over_2_5'].mean() + df['accel_decel_over_2_5'].std():
        risk += 0.3
    if row['peak_speed'] > df['peak_speed'].mean() + df['peak_speed'].std():
        risk += 0.2
    return min(risk, 1.0)
