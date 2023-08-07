import os
import re
from typing import Set

def normaliza_condition(string: str, seps: Set):
    return sum([True if sep in string else False for sep in seps])

def normalize_string(string: str, seps: Set = {" ", "\t", "\n"}) -> str:
    while normaliza_condition(string, seps) > 0:
        for sep in seps:
            if sep in string:
                string = string.strip().replace(sep, "")
        
    return string

if __name__=="__main__":
    string= """ test  test      
    test  """
    print(normalize_string(string))