# This line is importing the "calculator" function from another file.
# Imagine that "calculator" is like a tool or recipe that we've already written somewhere else,
# and now we are telling the computer, "Go and find that calculator tool for us."
# The "app" part is like a folder, and inside that folder, there's another file called "calculator.py",
# which has the tool (function) called "calculator" that we need.
from app.calculator import calculator

# This part of the code is super important! It checks if this file is being run directly by the computer.
# Let me explain: when we write Python programs, sometimes we want to run them directly,
# and other times we just want to use parts of the program inside other programs.
# The "__name__" is a special word in Python. It tells us if we are running the program directly.
# "__main__" is what Python calls this program when we run it directly.

# So, what this line means is: "If you're running this program directly (not as part of another program), 
# then go ahead and start the calculator."
if __name__ == "__main__":
    # Now, we use the calculator tool we got earlier. This will start the calculator, which is a program 
    # that keeps running and doing math based on what we tell it.
    calculator()
