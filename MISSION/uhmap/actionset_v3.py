import numpy as np
ActDigitLen = 100
def strActionToDigits(act_string):
    t = [ord(c) for c in act_string]
    d_len = len(t)
    assert d_len <= ActDigitLen
    pad = [-1 for _ in range(ActDigitLen-d_len)]
    return (t+pad)

def digitsToStrAction(digits):
    arr = [chr(d) for d in digits.astype(int) if d >= 0]
    if all([arr==0]):  
        return 'ActionSet3::N/A;N/A'
    else:
        return ''.join(arr)

"""
'ActionSet3::ChangeHeight;100'
"""