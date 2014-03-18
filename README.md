HMM
===

A simple implementation of the Baum-Welch algorithm for unsupervised learning of the parameters of an HMM.

This also implements the Cave-Neuwirth experiment. (In short, this is just treating the English language as a sequence of characters and learning HMM parameters from this sequence. There are some interesting insights).

The file called `pitest.py` does a similar thing, but with the digits of Pi.

This code is optimized for readability, not speed. Yes, it is slow. I know.

This relies on:
* matplotlib
* numpy
* a text file called `textdata.txt` (just text)

To run the Cave-Neuwirth experiment do:

    > python caveneuwirth.py
    
To run the Pi experiment, do:

    > python pitest.py
