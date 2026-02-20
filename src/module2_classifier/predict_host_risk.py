"""
Predict host risk using trained classifier.
"""

def predict_host_risk(model, features):
    """
    Predict host risk level.
    
    Args:
        model: Trained classifier model
        features: Input features
        
    Returns:
        Predicted risk level
    """
    prediction = model.predict(features)
    return prediction
