#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

""""create a 3x2 grid of subplots and set the figure title"""
plt.subplots(3, 2, figsize=(14, 6))
plt.suptitle('All in One', ha='center')
plt.subplots_adjust(top=0.92)
plt.subplots_adjust(wspace=0.3, hspace=0.5)

plt.subplot2grid((3, 3), (0, 0))
plt.plot(y0, 'r')

plt.subplot2grid((3, 3), (0, 1))
plt.scatter(x1, y1, s=1, color='magenta')
plt.xlabel('Height (in)', fontsize='x-small')
plt.ylabel('Weight (lbs)', fontsize='x-small')
plt.title("Men's Height vs Weight", fontsize='x-small')

plt.subplot2grid((3, 3), (1, 0))
plt.yscale("log")
plt.plot(y2, 'b')
plt.ylabel('Fraction Remaining', fontsize='x-small')
plt.xlabel('Time (years)', fontsize='x-small')
plt.title('Exponential Decay of C-14', fontsize='x-small')

plt.subplot2grid((3, 3), (1, 1))
plt.plot(x3, y31, '--', color='red', label='C-14')
plt.plot(x3, y32, color='green', label='Ra-226')
plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
plt.xlabel('Time (years)', fontsize='x-small')
plt.ylabel('Fraction Remaining', fontsize='x-small')
plt.title('Exponential Decay of Radioactive Elements', fontsize='x-small')

plt.subplot2grid((3, 3), (2, 0), colspan=2)
x = np.arange(0, 110, 10)
plt.hist(student_grades, x, edgecolor='black')
plt.xlabel('Grades', fontsize='x-small')
plt.ylabel('Number of Students', fontsize='x-small')
plt.title('Project A', fontsize='x-small')

plt.show()
