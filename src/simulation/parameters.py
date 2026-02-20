"""
Parameter management for SEIR simulations.
"""

class SEIRParameters:
    """Manages SEIR model parameters."""
    
    def __init__(self, beta=0.3, gamma=0.1, sigma=0.2):
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
    
    def update_beta(self, new_beta):
        """Update beta parameter."""
        self.beta = new_beta
    
    def to_dict(self):
        """Convert parameters to dictionary."""
        return {
            "beta": self.beta,
            "gamma": self.gamma,
            "sigma": self.sigma
        }
