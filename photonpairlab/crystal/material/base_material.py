import numpy as np
from scipy.misc import derivative

class BaseMaterial:
    """
    Base class for materials. Defines the interface for refractive index and group index calculations.
    """
    def is_biaxial(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def map_polarization_axis(
            self, 
            polarization_label # for uniaxial crystals: 'o', 'e'; for biaxial crystals 'x', 'y', 'z'
        ):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def refractive_index(
            self,
            wavelength, # Same for QPM and APM
            axis, # Same for QPM and APM
            temperature=None, # Used for QPM
            **kwargs # Additional parameters for future extensions
        ):
        raise NotImplementedError

    def effective_refractive_index(
            self,
            wavelength, # Same for QPM and APM
            theta_deg=None, # Used for APM
            phi_deg=None, # Used for APM
            **kwargs # Additional parameters for future extensions
        ):
        raise NotImplementedError

    def group_index(
            self,
            wavelength, # Same for QPM and APM
            axis=None, # Same for QPM and APM
            temperature=None, # Used for QPM
            theta_deg=None, # Used for APM
            phi_deg=None, # Used for APM
            **kwargs # Additional parameters for future extensions
        ):
        try:
            if theta_deg is not None:
                # Use effective refractive index for angle-based calculation
                n_func = lambda wl: self.effective_refractive_index(wl, theta_deg, phi_deg)
            elif axis is not None:
                # Use axis-based refractive index (for QPM along principal axis)
                n_func = lambda wl: self.refractive_index(wl, axis, temperature)
            else:
                raise ValueError("Either axis or theta_deg must be provided.")
            
            # Calculate the refractive index at the given wavelength
            n = n_func(wavelength)
            # Use numerical differentiation to calculate dn/dλ
            dn_dlambda = derivative(n_func, wavelength, dx=1e-9)
            
            # Calculate the group index
            group_index_value = n - wavelength * dn_dlambda
            
            return group_index_value
        except Exception as e:
            raise ValueError(f"Error in group_index: {e}")

    
    def thermal_expansion(
            self,
            length, # Same for QPM and APM
            axis, # Same for QPM and APM
            temperature=25, # Used for QPM
            **kwargs # Additional parameters for future extensions
        ):
        raise NotImplementedError