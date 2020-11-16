# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 12:58:13 2020

@author: Tianyang
"""

# -------------------------------------------------------------------------------------
# define raw SVI
class svi():
    # class object for SVI.
    # Input: "data": [n x 3], DataFrame; "newx": [n x 2], DataFrame
    # Parameters: "H": bandwidth of kernel
    # Output: "prediction": [n x 1], DataFrame
    def __init__(self,data):
        self.data = data