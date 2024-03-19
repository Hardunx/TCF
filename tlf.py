"""
 The MIT License (MIT)

Copyright © 2024 Hardunx

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and 
to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
    the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
from typing import List

# TLF stands for "Tokens Letters Frequency", which might give an idea of how it works.
class TLF:
    def __init__(self):
        pass
    
    def transform(self, X: List[str]) -> np.ndarray:
        """
        Transforms a list of strings into a numpy array of token letter frequencies.
        
        Parameters:
        X (List[str]): List of input strings.
        
        Returns:
        np.ndarray: Numpy array of token letter frequencies.
        """
        vectorized_inputs = []
        for example in X:
            token_letters_frequency = self._calculate_token_letters_frequency(example)
            vectorized_inputs.append(token_letters_frequency)
        return np.array(vectorized_inputs)

    def _calculate_token_letters_frequency(self, token: str) -> np.ndarray:
        """
        Calculates the letter frequencies of a given token.
        
        Parameters:
        token (str): Input token.
        
        Returns:
        np.ndarray: Numpy array of letter frequencies.
        """
        token = token.lower()

        valid_characters = [chr(i) for i in range(500)]

        letter_count = {char: 0 for char in valid_characters}
        for char in token:
            if char in letter_count:
                letter_count[char] += 1
        total_letters = sum(letter_count.values())
        frequencies = np.array([count / total_letters for count in letter_count.values()])
        return frequencies
