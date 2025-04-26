class Materials:
    """
    A class to encapsulate and manage material properties for nonlinear crystals.
    """
    def __init__(self):
        # Dictionary to store material properties
        self.materials = {
            "KTP": {
                "sellmeier": {
                    "y": {"A": 2.09930, "B": 0.922683, "C": 0.0467695, "D": 0.0138408},
                    "z": {"A": 2.12725, "B": 1.18431, "C": 0.0514852, "D": 0.6603, "E": 100.00507, "F": 0.00968956},
                },
                "temperature_corrections": {
                    "y": {
                        "n1": [6.28977e-6, 6.3061e-6, -6.0629e-6, 2.6486e-6],
                        "n2": [-0.14445e-8, 2.2244e-8, -3.5770e-8, 1.3470e-8],
                    },
                    "z": {
                        "n1": [9.9587e-6, 9.9228e-6, -8.9603e-6, 4.1010e-6],
                        "n2": [-1.1882e-8, 1.0459e-7, -9.8136e-8, 3.1481e-8],
                    },
                },
            },
            "BBO": {
                "sellmeier": {
                    "y": {"A": 2.7359, "B": 0.01878, "C": 0.01822, "D": -0.01516},
                    "z": {"A": 2.3753, "B": 0.01224, "C": 0.01667, "D": -0.01125},
                },
                # Add temperature corrections for BBO if available
            },
            # Add more materials here
        }

    def get_sellmeier_coefficients(self, material, axis):
        """
        Retrieve the Sellmeier coefficients for a given material and axis.

        Parameters:
        -----------
        material : str
            The name of the material (e.g., "KTP", "BBO").
        axis : str
            The axis for which to retrieve coefficients ("y" or "z").

        Returns:
        --------
        dict
            The Sellmeier coefficients for the specified material and axis.
        """
        try:
            return self.materials[material]["sellmeier"][axis]
        except KeyError:
            raise ValueError(f"Sellmeier coefficients for material '{material}' and axis '{axis}' not found.")

    def get_temperature_corrections(self, material, axis):
        """
        Retrieve the temperature correction coefficients for a given material and axis.

        Parameters:
        -----------
        material : str
            The name of the material (e.g., "KTP", "BBO").
        axis : str
            The axis for which to retrieve coefficients ("y" or "z").

        Returns:
        --------
        dict
            The temperature correction coefficients for the specified material and axis.
        """
        try:
            return self.materials[material]["temperature_corrections"][axis]
        except KeyError:
            return None  # Return None if no temperature corrections are available
        
    def get_thermo_optical_coefficients_UC(self, material, axis):
        """
        Retrieve the thermo-optical coefficients for a given material and axis.

        Parameters:
        -----------
        material : str
            The name of the material (e.g., "KTP", "BBO").
        axis : str
            The axis for which to retrieve coefficients ("x", "y" or "z").

        Returns:
        --------
        dict
            The temperature correction coefficients for the specified material and axis.
        """
        try:
            return self.materials[material]["thermo-optical-coefficients"][axis]
        except KeyError:
            return None  # Return None if no temperature corrections are available
