#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

names = ['Farrah', 'Fred', 'Felicia']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

# Create the stacked bar chart
bar_width = 0.5
x = np.arange(3)
for i in range(4):
    plt.bar(names, fruit[i], bottom=np.sum(fruit[:i], axis=0),
            label=['apples', 'bananas', 'oranges', 'peaches'][i],
            color=colors[i], width=bar_width)

# Add the legend
plt.legend()
plt.xticks(x)
plt.ylabel('Quantity of Fruit')
plt.title("Number of Fruit per Person")
plt.legend(loc="upper right")
plt.yticks(range(0, 81, 10))

plt.show()
