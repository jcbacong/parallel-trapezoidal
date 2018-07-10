import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("trapbench.csv")

## Execution Time
# labels = [list(df.columns)[i] for i in range(4)]
# y_axis = [np.log(df[i]) for i in labels]
# xreal_axis = [i for i in range(1,9)]
# fig, ax = plt.subplots()
# ax.plot(xreal_axis, y_axis[0], 'k',linestyle="-", label=labels[0])
# ax.plot(xreal_axis, y_axis[1], 'k',linestyle=":", label=labels[1])
# ax.plot(xreal_axis, y_axis[2], 'k',linestyle='-.', label=labels[2])
# ax.plot(xreal_axis, y_axis[3], 'k',linestyle="--", label=labels[3])
# ax.legend(loc='upper left', shadow=True)

# plt.ylabel("Log Execution Time")
# plt.xlabel(r"Iterations ($10^{x})$")


## Convergence
labels = [list(df.columns)[i] for i in range(4,8)]
y_axis = [df[i] for i in labels]
xreal_axis = [i for i in range(1,9)]
fig, ax = plt.subplots()
ax.plot(xreal_axis, y_axis[0], 'k',linestyle="-", label=labels[0])
ax.plot(xreal_axis, y_axis[1], 'k',linestyle=":", label=labels[1])
ax.plot(xreal_axis, y_axis[2], 'k',linestyle='-.', label=labels[2])
ax.plot(xreal_axis, y_axis[3], 'k',linestyle="--", label=labels[3])
ax.legend(loc='lower right', shadow=True)

plt.ylabel(r"$f(x) = \sin^{6}\left[\cos\left(x^{x}\right)+ x^{x}\sin(x^{x}) \right]$")
plt.xlabel(r"Iterations ($10^{x})$")





plt.show()