"""
 The MIT License (MIT)

Copyright © 2024 Hardunx

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and 
to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or 
    substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
from typing import List

# Tokens Characters Frequency
class TCF:
    @staticmethod
    def transform(inputs: List[str]):
        """
        Vectorizes a list of strings using Tokens Characters Frequency (TCF) approach.

        Args:
            inputs (List[str]): List of input strings to be vectorized.

        Returns:
            np.array: Array of TCF vectors representing the input strings.
        """
        vectors = []
        for input_str in inputs:
            # Initialize an array to store the count of characters
            total_chars = np.zeros(256)
            
            # Iterate through each token in the input string
            for token in input_str.split():
                # Initialize an array to store the count of characters in each token
                chars = np.zeros(256)
                
                # Count the occurrences of each character in the token
                for char in token:
                    chars[ord(char)] += 1
                
                # Accumulate the counts of characters in all tokens
                total_chars += chars
            
            # Calculate the Tokens Characters Frequency (TCF) for the input string
            tcf = total_chars / sum(total_chars)
            
            # Append the TCF vector for the input string to the list of vectors
            vectors.append(tcf)
        
        # Convert the list of vectors to a numpy array and return it
        return np.array(vectors)
