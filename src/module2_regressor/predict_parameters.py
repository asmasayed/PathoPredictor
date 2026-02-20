"""
Predict SEIR parameters using trained regressor.
"""

def predict_parameters(model, features):
    """
    Predict SEIR model parameters.
    
    Args:
        model: Trained regressor model
        features: Input features
        
    Returns:
        Dictionary of predicted parameters (beta, gamma, sigma)
    """
    predictions = model.predict(features)
    return {
        "beta": predictions[0],
        "gamma": predictions[1],
        "sigma": predictions[2]
    }
