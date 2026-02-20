"""
Predict beta adjustment using trained LSTM model.
"""

def predict_beta_adjustment(model, time_series_data):
    """
    Predict beta parameter adjustment from time series.
    
    Args:
        model: Trained LSTM model
        time_series_data: Time series input data
        
    Returns:
        Beta adjustment value
    """
    prediction = model.predict(time_series_data)
    return prediction
