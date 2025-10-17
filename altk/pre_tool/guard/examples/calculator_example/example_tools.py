from langchain_core.tools import tool


@tool
def add_tool(a: float, b: float) -> float:
    """
    Add two numbers.

    Parameters
    ----------
    a : float
        The first number to add.
    b : float
        The second number to add.

    Returns
    -------
    float
        The sum of a and b.
    """
    return a + b


@tool
def subtract_tool(c: float, d: float) -> float:
    """
    Subtract one number from another.

    Parameters
    ----------
    c : float
        The number to subtract from.
    d : float
        The number to subtract.

    Returns
    -------
    float
        The result of a minus b.
    """
    return c - d


@tool
def multiply_tool(e: float, f: float) -> float:
    """
    Multiply two numbers.

    Parameters
    ----------
    e : float
        The first number.
    f : float
        The second number.

    Returns
    -------
    float
        The product of a and b.
    """
    return e * f


@tool
def divide_tool(g: float, h: float) -> float:
    """
    Divide one number by another.

    Parameters
    ----------
    g : float
        The dividend.
    h : float
        The divisor (must not be zero).

    Returns
    -------
    float
        The result of a divided by b.
    """
    return g / h
