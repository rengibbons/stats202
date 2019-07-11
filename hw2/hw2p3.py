"""Performs analysis for STAT 202 HW 2 P 3, Sum 2019"""
import numpy as np

def logistic_function(beta, student):
    """Computes probability of student getting an A."""
    b_0 = beta[0]
    b_1 = beta[1]
    b_2 = beta[2]

    hrs = student[0]
    gpa = student[1]

    prob = np.exp(b_0 + b_1 * hrs + b_2 * gpa) / (1 + np.exp(b_0 + b_1 * hrs + b_2 * gpa))
    return prob

def logit_function(beta, student, prob):
    """"Computes study hours needed for 50% chance of getting an A."""
    b_0 = beta[0]
    b_1 = beta[1]
    b_2 = beta[2]

    gpa = student[1]

    hrs = (np.log(prob / (1 - prob)) - b_0 - b_2 * gpa) / b_1
    return hrs


def main():
    """"Performs simple computations for a logistic regression model."""
    beta = [-6, 0.05, 1]
    student = [40, 3.5]

    part_a = logistic_function(beta, student)
    print('probability of getting an A: {:.3f}%'.format(100*part_a))

    prob = 0.5
    part_b = logit_function(beta, student, prob)
    print('study hours need for 50% chance of getting an A: {:.3f}'.format(part_b))

if __name__ == '__main__':
    main()
