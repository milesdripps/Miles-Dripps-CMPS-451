# Miles Dripps, CMPS 451: Regression Assignment
# Confirm the following using regression:
#
#(i) “age” and “bmi”
#
#(2) “age” and “charges”
#
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from tensorflow.python.ops.losses.losses_impl import mean_squared_error

#
#   As the results show, both Age vs Bmi and Age vs Charges
#   have an R squared value of less than .1, meaning there
#   is almost no relationship between the two.
#




def getInfo():
    #Put the age, bmi, and charges into arrays
    with open('insurance.csv', mode='r') as csvfile:
        age = []
        bmi = []
        charges = []
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            age.append(float(row[0]))
            bmi.append(float(row[2]))
            charges.append(float(row[6]))
    return np.array(age), np.array(bmi), np.array(charges)

def age_bmi(age,bmi):
    age = age
    bmi = bmi
    #Get slope and intercept
    slope, intercept = np.polyfit(age, bmi, 1)

    bmi_prediction = slope * age + intercept

    r_sqr = r2_score(bmi, bmi_prediction)
    #plot
    plt.scatter(age, bmi, color='blue')
    plt.plot(age, bmi_prediction, color='red',
             label=f'Line: y={slope:.2f}x + {intercept:.2f}')

    plt.xlabel('Age')
    plt.ylabel('Bmi')
    plt.xlim(18, 60)
    plt.ylim(0, 60)
    plt.title('Age vs Bmi: Regression')
    print(f"# # # # Age vs BMI # # # #\nR-Squared:{r_sqr:.3f}\nSlope:{slope:.2f}")
    plt.show()


def age_charges(age,charges):
    age = age
    charges = charges
    slope, intercept = np.polyfit(age, charges, 1)

    charges_prediction = slope * age + intercept

    r_sqr = r2_score(charges, charges_prediction)

    plt.scatter(age, charges, color='blue')
    plt.plot(age, charges_prediction, color='red',
             label=f'Line: y={slope:.2f}x + {intercept:.2f}')

    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.title('Age vs Charges: Regression')
    print(f"# # # # Age vs BMI # # # #\nR-Squared:{r_sqr:.3f}\nSlope:{slope:.2f}")
    plt.show()

def main():
    #Get age, bmi, charges values
    age, bmi, charges = getInfo()
    #Function to see the regression between age and bmi
    age_bmi(age,bmi)
    #Function to see the regression between age and charges
    age_charges(age,charges)


main()



