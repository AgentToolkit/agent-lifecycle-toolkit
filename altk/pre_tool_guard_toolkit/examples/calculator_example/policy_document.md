# Calculator Usage Policy

This document outlines the rules and constraints that govern the behavior and usage of the calculator application.

## General Principles

- The calculator should perform accurate and reliable computations.
- All operations must conform to mathematical standards and avoid undefined behavior.

## Supported Operations

The calculator supports the following operations:

- Addition (`+`)
- Subtraction (`-`)
- Multiplication (`*`)
- Division (`/`)

## Operation Constraints

- **Division by Zero is Not Allowed**
  The calculator **must not** allow division by zero.
  If a user attempts to divide by zero, the operation must be rejected and an appropriate error message should be shown
  (e.g., `"Error: Division by zero is not allowed."`).

- **Summing Numbers Whose Product is 365 is Not Allowed**
  The calculator **must not** allow addition of two or more numbers if their multiplication result equals `365`.
  For example, adding `5 + 73`  should be disallowed, because `5 * 73 = 365` .
  In such cases, the operation must be rejected with an error like:
  `"Error: Addition of numbers whose product equals 365 is not allowed."`
