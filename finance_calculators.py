"""
Refer to the Pseudocode File (CapstoneProject1_Pseudocode.txt) for comprehensive details on the program's functionality
Task 5.1 Capstone Project - Variables and Control Structures file (finance_calculators.py)
"""
# Initialise variables
import math
fin_calc_option = ['investment', 'bond']    # List the valid options investment or bond for the user

# Program starts: Ask user for inputs
back_to_start = True    # Boolean flag to control the program flow
# Use of while loop with Try / Except statement
while back_to_start:
    try:
        user_selection = input(f"""\nWelcome to Retail Banking 101!\n
Please review the Options below and instructions on how to proceed.

Option 1: Investment - to calculate the amount of interest you will earn on your investment
Option 2: Bond - to calculate the amount you will have to pay on a home loan

Please type your choice of either 'Investment' or 'Bond' to proceed: """).lower()    # Make all text inputs lower case
# Take in users choice of input and follow next steps
        if user_selection in fin_calc_option:
            back_to_start = False    # Break using Boolean control
        else:
            print("\nInvalid input. Please select an option by entering either 'investment' or 'bond'.\n")           
        if user_selection == 'investment':    # User selected investment
            P = float(input("Enter the amount of money you wish to deposit (in GBP): "))
            r = float(input("Enter the expected interest rate percentage (Input example: 5.00 Represents 5%): ")) / 100
            t = int(input("Enter the number of years you plan on investing: "))
            interest = input("Would you like the interest rate to be calculated as 'simple' or 'compound' interest? ").lower()    # Make all text inputs lower case
            # Calculate the interest 'simple' or 'compound'
            if interest == 'simple':    # User chose simple interest formula
                A = P * (1 + r * t)
                print(f"The total amount to be returned (i.e. initial investment and interest) following your initial investment of £{P:.2f} at an interest rate ({interest}) of {r*100}% after {t} years will be: £{A:.2f}")
            elif interest == 'compound':
                A = P * math.pow((1 + r), t)    # User chose compound interest formula (which uses to the power of)
                print(f"The total amount to be returned (i.e. initial investment and interest) following your initial investment of £{P:.2f} at an interest rate ({interest}) of {r*100}% after {t} years will be: £{A:.2f}")
            else:
                print("\nInvalid input. Next time, please enter either 'simple' or 'compound'.\n")
                back_to_start = True # If a user inputs an invalid response the program will restart from the beginning
        elif user_selection == 'bond': # User selected bond
            P = float(input("Enter the present value (current market value) of the house (in GBP): "))
            i_user_input = float(input("Enter the expected annual interest rate percentage (Input example: 5.00 Represents 5%): "))
            n = int(input("Enter the number of months in which the bond will be repaid: "))
            i = (i_user_input / 100) / 12
            repayment = (i * P) / (1 - math.pow((1 + i), -n))    # One option so one formula (uses to the power of)
            print(f"The monthly interest and capital, bond repayment amount will be: £{repayment:.2f}")
    # Handle all user input errors with use of exception   
    except Exception as e:
        print(f"\nProgram has ended, please start again, be sure to use valid inputs as requested and no symbols e.g. (%, £ or GBP, $, &), {e}\n")
        back_to_start = True # If a user inputs an invalid response or a symbol the program will restart from the beginning
# Program ends