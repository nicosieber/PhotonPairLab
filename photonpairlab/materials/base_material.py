import numpy as np
from scipy.misc import derivative

class BaseMaterial:
    """
    Base class for materials. Defines the interface for refractive index and group index calculations.
    """  
    def refractive_index(self, wavelength, axis, temperature=None):
        """
        Calculate the refractive index for a given wavelength, axis and temperature.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def group_index(self, wavelength, axis, temperature=25):
        try:
            n_func = lambda wavelength: self.refractive_index(wavelength, axis, temperature)
            
            # Calculate the refractive index at the given wavelength
            n = n_func(wavelength)
            # Use numerical differentiation to calculate dn/dλ
            dn_dlambda = derivative(n_func, wavelength, dx=1e-9)
            
            # Calculate the group index
            group_index_value = n - wavelength * dn_dlambda
            
            return group_index_value

        except Exception as e:
            raise ValueError(f"Error in group_index: {e}")  
    
    #@abstractmethod
    def thermal_expansion(self, length, axis, temperature=25):
        """
        Calculate the thermal expansion for a given length and temperature using thermal expansion coefficients.
        This is a placeholder method and should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")