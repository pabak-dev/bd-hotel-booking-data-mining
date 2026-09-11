import pandas as pd
import numpy as np

# =========================================================
# CONFIG
# =========================================================
INPUT_FILE = "hotel_bookings.csv"
N_ROWS = 12000
SEED = 42

rng = np.random.default_rng(SEED)

# =========================================================
# 1. LOAD + SAMPLE ORIGINAL DATA
# =========================================================
df = pd.read_csv(INPUT_FILE, na_values=["NULL"])

# Random sample from original dataset.
# This preserves the original relationships between
# lead_time, market_segment, cancellation, etc.
df = df.sample(n=N_ROWS, random_state=SEED).reset_index(drop=True)

# Give every synthetic booking its own ID
df.insert(0, "booking_id", [f"BDH_{i:05d}" for i in range(1, N_ROWS + 1)])


# =========================================================
# 2. BANGLADESH HOTEL LOCATIONS
# =========================================================

# Location depends somewhat on hotel type.
city_locations = [
    "Dhaka",
    "Chattogram",
    "Sylhet",
    "Rajshahi",
    "Khulna"
]

city_weights = [0.55, 0.20, 0.12, 0.07, 0.06]

resort_locations = [
    "Coxs_Bazar",
    "Sylhet",
    "Kuakata",
    "Sreemangal",
    "Bandarban"
]

resort_weights = [0.50, 0.14, 0.13, 0.13, 0.10]


def generate_location(hotel):
    if hotel == "City Hotel":
        return rng.choice(city_locations, p=city_weights)
    else:
        return rng.choice(resort_locations, p=resort_weights)


df["hotel_location"] = df["hotel"].apply(generate_location)


# =========================================================
# 3. LOCALIZE HOTEL TYPE NAMES
# =========================================================
df["hotel"] = df["hotel"].replace({
    "City Hotel": "City Hotel",
    "Resort Hotel": "Resort"
})


# =========================================================
# 4. GUEST COUNTRY / ORIGIN
# =========================================================
# Mostly Bangladeshi guests, with some international guests.

countries = [
    "BGD",
    "IND",
    "GBR",
    "USA",
    "CAN",
    "ARE",
    "SAU",
    "MYS",
    "SGP",
    "AUS"
]

country_weights = [
    0.78,  # Bangladesh
    0.06,
    0.035,
    0.03,
    0.02,
    0.025,
    0.015,
    0.015,
    0.01,
    0.01
]

df["country"] = rng.choice(
    countries,
    size=len(df),
    p=country_weights
)


# =========================================================
# 5. LOCALIZE ADR TO BANGLADESHI TAKA
# =========================================================
# We preserve the general relative pricing information from
# the original ADR instead of generating totally random prices.

original_adr = pd.to_numeric(df["adr"], errors="coerce")

# Remove impossible/extreme values before transformation
original_adr = original_adr.clip(lower=20, upper=400)

# Convert the scale into a plausible synthetic BDT range.
# This is NOT intended to represent actual market statistics.
normalized = (original_adr - 20) / (400 - 20)

df["adr"] = 2500 + normalized * 12500

# Location adjustment
location_multiplier = {
    "Dhaka": 1.20,
    "Chattogram": 1.05,
    "Sylhet": 1.05,
    "Rajshahi": 0.85,
    "Khulna": 0.85,
    "Coxs_Bazar": 1.25,
    "Kuakata": 0.90,
    "Sreemangal": 1.00,
    "Bandarban": 0.95,
}

df["adr"] *= df["hotel_location"].map(location_multiplier)

# Small random variation
df["adr"] *= rng.normal(1.0, 0.08, size=len(df))

# Keep sensible values
df["adr"] = df["adr"].clip(1800, 25000).round(0)


# =========================================================
# 6. PAYMENT METHOD - NEW BANGLADESH-RELEVANT ATTRIBUTE
# =========================================================

payment_methods = [
    "Mobile Banking",
    "Card",
    "Cash",
    "Bank Transfer"
]

payment_weights = [0.42, 0.30, 0.16, 0.12]

df["payment_method"] = rng.choice(
    payment_methods,
    size=len(df),
    p=payment_weights
)


# =========================================================
# 7. LOCALIZE ARRIVAL YEAR
# =========================================================
# Keep month/day/week structure from original rows,
# but make the records look like a more recent synthetic set.

df["arrival_date_year"] = rng.choice(
    [2024, 2025],
    size=len(df),
    p=[0.45, 0.55]
)


# =========================================================
# 8. ADD A SMALL AMOUNT OF MISSING DATA
# =========================================================
# Gives you something meaningful to handle using
# ReplaceMissingValues in WEKA.

def introduce_missing(column, fraction):
    indices = rng.choice(
        df.index,
        size=int(len(df) * fraction),
        replace=False
    )
    df.loc[indices, column] = np.nan


introduce_missing("children", 0.015)
introduce_missing("country", 0.010)
introduce_missing("payment_method", 0.010)


# =========================================================
# 9. BASIC VALIDITY FIXES
# =========================================================

# At least one adult
df.loc[df["adults"] < 1, "adults"] = 1

# No negative counts
count_columns = [
    "children",
    "babies",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "required_car_parking_spaces",
    "total_of_special_requests"
]

for col in count_columns:
    df.loc[df[col] < 0, col] = 0


# Make sure every booking has at least one night
zero_stay = (
    (df["stays_in_weekend_nights"] == 0) &
    (df["stays_in_week_nights"] == 0)
)

df.loc[zero_stay, "stays_in_week_nights"] = 1


# =========================================================
# 10. SAVE FULL SYNTHETIC DATASET
# =========================================================

df.to_csv(
    "bd_hotel_bookings_raw.csv",
    index=False
)


# =========================================================
# 11. CREATE ML VERSION
# =========================================================
# We DO NOT delete these from the raw dataset.
# We only remove them from the version intended for ML.

ml_df = df.copy()

remove_columns = [
    "booking_id",              # identifier
    "reservation_status",      # target leakage
    "reservation_status_date", # target leakage
    "company",                 # mostly missing / identifier
    "agent",                   # identifier-like
    "assigned_room_type"       # may be known after booking
]

ml_df = ml_df.drop(
    columns=[c for c in remove_columns if c in ml_df.columns]
)

ml_df.to_csv(
    "bd_hotel_bookings_ml.csv",
    index=False
)


# =========================================================
# 12. QUICK SUMMARY
# =========================================================

print("\n=== DATASET CREATED ===")
print("Rows:", len(df))
print("Columns (raw):", len(df.columns))
print("Columns (ML):", len(ml_df.columns))

print("\nCancellation distribution:")
print(df["is_canceled"].value_counts())
print(df["is_canceled"].value_counts(normalize=True).round(3))

print("\nHotel locations:")
print(df["hotel_location"].value_counts())

print("\nCountries:")
print(df["country"].value_counts(dropna=False))

print("\nADR BDT:")
print(df["adr"].describe().round(2))

print("\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nSaved:")
print("  bd_hotel_bookings_raw.csv")
print("  bd_hotel_bookings_ml.csv")