import matplotlib.pyplot as plt
import pickle

with open("data_50gens", "rb") as f:
	logbook = pickle.load(f)

gen = logbook.select("gen")
max_foods = logbook.chapters["food"].select("max")
max_fitness = logbook.chapters["fitness"].select("max")
avg_foods = logbook.chapters["food"].select("avg")
avg_fitness = logbook.chapters["fitness"].select("avg")


# fit_mins = logbook.chapters["fitness"].select("min")
# size_avgs = logbook.chapters["size"].select("avg")
#
#
fig, ax1 = plt.subplots()
line1 = ax1.plot(gen, max_fitness, "b-", label="Maximum Fitness")
line2 = ax1.plot(gen, avg_fitness, "b--", label="Average Fitness")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness", color="b")
for tl in ax1.get_yticklabels():
    tl.set_color("b")

ax2 = ax1.twinx()
line3 = ax2.plot(gen, max_foods, "r-", label="Maximum Food")
line4 = ax2.plot(gen, avg_foods, "r--", label="Average Food")
ax2.set_ylabel("Size", color="r")
for tl in ax2.get_yticklabels():
    tl.set_color("r")

lns = line1 + line2 + line3 + line4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="best")

plt.xticks(ticks=list(range(0, 51, 5)))
plt.show()
