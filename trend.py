def trend(df):
    return {
        "temp": df["temp"].mean(),
        "precip": df["precip"].mean(),
        "wind": df["wind"].mean(),
        "pressure": df["pressure"].mean()
    }
