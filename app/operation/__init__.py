# operations.py

class Operation:
    """
    The Operation class encapsulates basic arithmetic operations as static methods.
    This design groups related functions (addition, subtraction, multiplication, and division) 
    in a single class, making the code more modular and organized.

    **Object-Oriented Programming (OOP) Principles Illustrated:**
    - **Encapsulation:** This class groups all arithmetic operations together, making it easier 
      to maintain, test, and reuse these methods in other parts of the code.
    - **Abstraction:** Users of this class only need to know the function names and purpose, 
      not how they work internally.
    - **Reusability:** Static methods can be called directly on the class without creating an instance, 
      making it straightforward to reuse these methods anywhere.
    - **Organization:** By placing all basic operations in a single class, the code becomes 
      more structured and readable.
    
    **Why Static Methods?**
    - **Statelessness**: These methods do not rely on any instance-specific data; they only 
      depend on input parameters. Static methods are ideal for utility functions that 
      perform independent operations.
    - **Ease of Access**: Because we don’t need to create an instance of Operation to call 
      these methods, it’s easy to use them across different parts of the program.
    """

    @staticmethod
    def addition(a: float, b: float) -> float:
        """
        Adds two floating-point numbers and returns the result.

        **Parameters:**
        - `a (float)`: The first number to add.
        - `b (float)`: The second number to add.
        
        **Returns:**
        - `float`: The sum of `a` and `b`.

        **Example:**
        >>> Operation.addition(5.0, 3.0)
        8.0

        **Why Use Static Method for Addition?**
        - Static methods are suitable for functions like addition because they are 
          independent of any instance-specific data, relying only on the parameters.
        """
        return a + b  # Performs addition of two numbers and returns the result.
    
    @staticmethod
    def subtraction(a: float, b: float) -> float:
        """
        Subtracts the second floating-point number from the first and returns the result.

        **Parameters:**
        - `a (float)`: The number from which to subtract.
        - `b (float)`: The number to subtract.
        
        **Returns:**
        - `float`: The difference between `a` and `b`.

        **Example:**
        >>> Operation.subtraction(10.0, 4.0)
        6.0

        **Design Choice: Why Separate Functions for Each Operation?**
        - By having separate functions, we make each operation clear and isolated, 
          adhering to the **Single Responsibility Principle (SRP)**. Each function 
          handles one specific task (addition, subtraction, etc.), making it easier 
          to test and modify them independently.
        """
        return a - b  # Subtracts the second number from the first and returns the difference.
    
    @staticmethod
    def multiplication(a: float, b: float) -> float:
        """
        Multiplies two floating-point numbers and returns the product.

        **Parameters:**
        - `a (float)`: The first number to multiply.
        - `b (float)`: The second number to multiply.
        
        **Returns:**
        - `float`: The product of `a` and `b`.

        **Example:**
        >>> Operation.multiplication(2.0, 3.0)
        6.0

        **Advantages of Static Methods in Utility Classes:**
        - Static methods in utility classes like this one provide simple access to functions 
          without requiring an instance of the class. This reduces overhead and makes 
          the methods easily reusable in other parts of the program.
        """
        return a * b  # Multiplies the two numbers and returns the product.
    
    @staticmethod
    def division(a: float, b: float) -> float:
        """
        Divides the first floating-point number by the second and returns the quotient.

        **Parameters:**
        - `a (float)`: The dividend.
        - `b (float)`: The divisor.
        
        **Returns:**
        - `float`: The quotient of `a` divided by `b`.

        **Raises:**
        - `ValueError`: If the divisor `b` is zero, as division by zero is undefined.

        **Example:**
        >>> Operation.division(10.0, 2.0)
        5.0
        >>> Operation.division(10.0, 0.0)
        Traceback (most recent call last):
            ...
        ValueError: Division by zero is not allowed.

        **Error Handling:**
        - Division requires extra error handling to prevent division by zero, which 
          would cause a runtime error. Here, we check if `b` is zero and raise a 
          `ValueError` with a descriptive message if it is.
        
        **Design Insight: Why Raise an Error for Division by Zero?**
        - Raising an error in this case is a **Defensive Programming** technique, 
          helping us prevent unexpected results. Instead of letting the program fail 
          silently or crash, we handle the error gracefully, ensuring that any part of 
          the program using this function will be alerted to the issue.
        """
        if b == 0:
            # Checks if the divisor is zero to prevent undefined division.
            raise ValueError("Division by zero is not allowed.")  # Raises an error if division by zero is attempted.
        return a / b  # Divides `a` by `b` and returns the quotient.
    @staticmethod
    def power(a: float, b: float) -> float:
        return a ** b 
    @staticmethod
    def modulus(a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Modulus by zero is not allowed.")
        return a % b