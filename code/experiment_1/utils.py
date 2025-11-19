import os

def find_assistant_token_end(ls):
    for i in range(len(ls)-2):
        if ls[i] == 128006 and ls[i+1] == 78191 and ls[i+2] == 128007:
            return i + 3
        
    return -1    