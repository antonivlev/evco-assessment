# This code defines the agent (as in the playable version) in a way that can be called and executed from an evolutionary algorithm. The code is partial and will not execute. You need to add to the code to create an evolutionary algorithm that evolves and executes a snake agent.
import curses
import random
import operator
import numpy as np
import pygraphviz as pgv

from math import sqrt
from functools import partial
from deap import gp
from deap import algorithms
from deap import base
from deap import creator
from deap import tools



S_RIGHT, S_LEFT, S_UP, S_DOWN = 0,1,2,3
XSIZE,YSIZE = 14,14
NFOOD = 1 # NOTE: YOU MAY NEED TO ADD A CHECK THAT THERE ARE ENOUGH SPACES LEFT FOR THE FOOD (IF THE TAIL IS VERY LONG)

def if_then_else(condition, out1, out2):
	out1() if condition() else out2()


# This class can be used to create a basic player object (snake agent)
class SnakePlayer(list):
	global S_RIGHT, S_LEFT, S_UP, S_DOWN
	global XSIZE, YSIZE

	def __init__(self):
		self.direction = S_RIGHT
		self.body = [ [4,10], [4,9], [4,8], [4,7], [4,6], [4,5], [4,4], [4,3], [4,2], [4,1],[4,0] ]
		self.score = 1
		self.ahead = []
		self.food = []

	def _reset(self):
		self.direction = S_RIGHT
		self.body[:] = [ [4,10], [4,9], [4,8], [4,7], [4,6], [4,5], [4,4], [4,3], [4,2], [4,1],[4,0] ]
		self.score = 1
		self.ahead = []
		self.food = []

	def getAheadLocation(self):
		self.ahead = [ self.body[0][0] + (self.direction == S_DOWN and 1) + (self.direction == S_UP and -1), self.body[0][1] + (self.direction == S_LEFT and -1) + (self.direction == S_RIGHT and 1)]

	def updatePosition(self):
		self.getAheadLocation()
		self.body.insert(0, self.ahead )

	## You are free to define more sensing options to the snake

	def snakeHasCollided(self):
		self.hit = False
		if self.body[0][0] == 0 or self.body[0][0] == (YSIZE-1) or self.body[0][1] == 0 or self.body[0][1] == (XSIZE-1): self.hit = True
		if self.body[0] in self.body[1:]: self.hit = True
		return( self.hit )


	# MOVEMENT TERMINALS
	def changeDirectionUp(self):
		self.direction = S_UP

	def changeDirectionRight(self):
		self.direction = S_RIGHT

	def changeDirectionDown(self):
		self.direction = S_DOWN

	def changeDirectionLeft(self):
		self.direction = S_LEFT


	# SENSING PRIMITIVES
	#
	def if_right_gets_closer_to_food(self, out1, out2):
		return partial(
			if_then_else,
			partial(self.does_dir_get_closer_to_food, S_RIGHT),
			out1,
			out2
		)
	def if_left_gets_closer_to_food(self, out1, out2):
		return partial(
			if_then_else,
			partial(self.does_dir_get_closer_to_food, S_LEFT),
			out1,
			out2
		)
	def if_up_gets_closer_to_food(self, out1, out2):
		return partial(
			if_then_else,
			partial(self.does_dir_get_closer_to_food, S_UP),
			out1,
			out2
		)
	def if_down_gets_closer_to_food(self, out1, out2):
		return partial(
			if_then_else,
			partial(self.does_dir_get_closer_to_food, S_DOWN),
			out1,
			out2
		)


	# food
	def if_food_ahead(self, out1, out2):
		return partial(if_then_else, self.sense_food_ahead, out1, out2)


	# wall avoidance
	def if_wall_ahead(self, out1, out2):
		return partial(if_then_else, self.sense_wall_ahead, out1, out2)


	# tail avoidance
	def if_tail_ahead(self, out1, out2):
		return partial(if_then_else, self.sense_tail_ahead, out1, out2)

	# tail and wall avoidance
	def if_death_left(self, out1, out2):
		return partial(if_then_else, lambda: self.does_dir_kill(S_LEFT), out1, out2)

	def if_death_right(self, out1, out2):
		return partial(if_then_else, lambda: self.does_dir_kill(S_RIGHT), out1, out2)

	def if_death_up(self, out1, out2):
		return partial(if_then_else, lambda: self.does_dir_kill(S_UP), out1, out2)

	def if_death_down(self, out1, out2):
		return partial(if_then_else, lambda: self.does_dir_kill(S_DOWN), out1, out2)


	# hmm
	def if_moving_up(self, out1, out2):
		return partial(if_then_else, lambda: self.direction == S_UP, out1, out2)

	def if_moving_down(self, out1, out2):
		return partial(if_then_else, lambda: self.direction == S_DOWN, out1, out2)

	def if_moving_left(self, out1, out2):
		return partial(if_then_else, lambda: self.direction == S_LEFT, out1, out2)

	def if_moving_right(self, out1, out2):
		return partial(if_then_else, lambda: self.direction == S_RIGHT, out1, out2)


	# SENSING HELPERS
	def does_dir_get_closer_to_food(self, dir):
		new_ahead = [
			self.body[0][0] +
			(dir == S_DOWN and 1) +
			(dir == S_UP and -1),
			self.body[0][1] +
			(dir == S_LEFT and -1) +
			(dir == S_RIGHT and 1)
		]
		if self.ahead != []:
			return distToFood(new_ahead, self) <= distToFood(self.ahead, self)
		else:
			return False

	def does_dir_kill(self, dir):
		new_ahead = [
			self.body[0][0] +
			(dir == S_DOWN and 1) +
			(dir == S_UP and -1),
			self.body[0][1] +
			(dir == S_LEFT and -1) +
			(dir == S_RIGHT and 1)
		]
		# 		tail 					or 	wall collision
		return (new_ahead in self.body) or (new_ahead == [])

	def sense_wall_ahead(self):
		self.getAheadLocation()
		return( self.ahead[0] == 0 or self.ahead[0] == (YSIZE-1) or self.ahead[1] == 0 or self.ahead[1] == (XSIZE-1) )

	def sense_food_ahead(self):
		self.getAheadLocation()
		return self.ahead in self.food

	def sense_tail_ahead(self):
		self.getAheadLocation()
		return self.ahead in self.body

# This function places a food item in the environment
def placeFood(snake):
	food = []
	while len(food) < NFOOD:
		potentialfood = [random.randint(1, (YSIZE-2)), random.randint(1, (XSIZE-2))]
		if not (potentialfood in snake.body) and not (potentialfood in food):
			food.append(potentialfood)
	snake.food = food  # let the snake know where the food is
	return( food )


snake = SnakePlayer()

def evaluateSnakeStrategy(individual):
	maxSeed = 0
	maxFitness = 0
	fitnesses = []
	foods = []
	for i in range(4):
		# seeded run of the game
		seed = int(random.random()*100)
		random.seed(seed)
		fitness = runGame(individual)[0]
		foods.append(individual.food_eaten)
		fitnesses.append(fitness)
		if fitness > maxFitness:
			maxFitness = fitness
			maxSeed = seed

	individual.seed = maxSeed
	individual.avg_food = np.mean(foods)
	var_foods = np.var(foods)
	return np.mean(fitnesses), individual.avg_food/(var_foods + 1)

pset = gp.PrimitiveSet("main", 0)

#followPath()
#pset.addPrimitive(snake.if_food_ahead, 2)
# pset.addPrimitive(snake.if_wall_ahead, 2)
# pset.addPrimitive(snake.if_tail_ahead, 2)
pset.addPrimitive(snake.if_death_left, 2)
pset.addPrimitive(snake.if_death_right, 2)
pset.addPrimitive(snake.if_death_up, 2)
pset.addPrimitive(snake.if_death_down, 2)
# pset.addPrimitive(snake.if_moving_up, 2)
# pset.addPrimitive(snake.if_moving_down, 2)
# pset.addPrimitive(snake.if_moving_left, 2)
# pset.addPrimitive(snake.if_moving_right, 2)
pset.addPrimitive(snake.if_right_gets_closer_to_food, 2)
pset.addPrimitive(snake.if_left_gets_closer_to_food, 2)
pset.addPrimitive(snake.if_up_gets_closer_to_food, 2)
pset.addPrimitive(snake.if_down_gets_closer_to_food, 2)

pset.addTerminal(snake.changeDirectionUp)
pset.addTerminal(snake.changeDirectionDown)
pset.addTerminal(snake.changeDirectionLeft)
pset.addTerminal(snake.changeDirectionRight)

creator.create("FitnessFunc", base.Fitness, weights=(1.0,0.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessFunc)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=5)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("evaluate", evaluateSnakeStrategy)

# This outline function is the same as runGame (see below). However,
# it displays the game graphically and thus runs slower
# This function is designed for you to be able to view and assess
# your strategies, rather than use during the course of evolution
def displayStrategyRun(individual):
	global snake
	global pset

	routine = gp.compile(individual, pset)

	curses.initscr()
	win = curses.newwin(YSIZE, XSIZE, 0+5, 0)
	win.keypad(1)
	curses.noecho()
	curses.curs_set(0)
	win.border(0)
	win.nodelay(1)
	win.timeout(120)

	snake._reset()
	food = placeFood(snake)

	for f in food:
		win.addch(f[0], f[1], '@')

	timer = 0
	collided = False
	while not collided and not timer == ((2*XSIZE) * YSIZE):

		# Set up the display
		win.border(0)
		win.addstr(0, 2, 'Score : ' + str(snake.score) + ' ')
		# for possible debugging later
		#win.addstr(1, 2, 'Fitns : ' + str(individual.runningFitness) + ' ')

		win.getch()

		## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##
		updateAction = gp.compile(individual, pset)
		updateAction()

		snake.updatePosition()

		if snake.body[0] in food:
			snake.score += 1
			for f in food: win.addch(f[0], f[1], ' ')
			food = placeFood(snake)
			for f in food: win.addch(f[0], f[1], '@')
			timer = 0
		else:
			last = snake.body.pop()
			win.addch(last[0], last[1], ' ')
			timer += 1 # timesteps since last eaten
		win.addch(snake.body[0][0], snake.body[0][1], 'o')

		collided = snake.snakeHasCollided()
		hitBounds = (timer == ((2*XSIZE) * YSIZE))

	curses.endwin()

	print(collided)
	print(hitBounds)
	input("Press to continue...")

	return snake.score,



def distToFood(ahead, snake):
	return sqrt((ahead[0] - snake.food[0][0])**2 + (ahead[1] - snake.food[0][1])**2)

# This outline function provides partial code for running the game with an evolved agent
# There is no graphical output, and it runs rapidly, making it ideal for
# you need to modify it for running your agents through the game for evaluation
# which will depend on what type of EA you have used, etc.
# Feel free to make any necessary modifications to this section.
def runGame(individual):
	global snake
	individual.food_eaten = 0

	totalScore = 0

	snake._reset()
	food = placeFood(snake)
	timer = 0
	while not snake.snakeHasCollided() and not timer == XSIZE * YSIZE:

		## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##
		updateAction = gp.compile(individual, pset)
		updateAction()

		snake.updatePosition()

		if snake.body[0] in food:
			individual.food_eaten += 1
			#snake.score += 1
			food = placeFood(snake)
			timer = 0
		else:
			snake.body.pop()
			timer += 1 # timesteps since last eaten

		totalScore += snake.score - timer*0.08
		# print("-- timer:", timer)

	return totalScore,


hof = tools.HallOfFame(1)

def main():
	global snake
	global pset
	pop = toolbox.population(n=5000)
	stats_fit  = tools.Statistics(lambda ind: ind.fitness.values[0])
	stats_food = tools.Statistics(lambda ind: ind.avg_food)
	mstats = tools.MultiStatistics(fitness=stats_fit, food=stats_food)
	mstats.register("avg", np.mean)
	mstats.register("std", np.std)
	mstats.register("min", np.min)
	mstats.register("max", np.max)

	pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.4, ngen=100,
									stats=mstats, halloffame=hof, verbose=True)


def seeBest():
	print(str(hof[0]))
	random.seed(hof[0].seed)
	drawTree(hof[0])
	displayStrategyRun(hof[0])

def testBest():
	print(str(hof[0]))
	displayStrategyRun(hof[0])

def drawTree(expr):
	nodes, edges, labels = gp.graph(expr)

	# Plot the tree
	g = pgv.AGraph(nodesep=1.0)
	g.add_nodes_from(nodes)
	g.add_edges_from(edges)
	g.layout(prog="dot")

	for i in nodes:
		n = g.get_node(i)
		n.attr["label"] = labels[i]

	g.draw("tree.pdf")



main()

# expr = gp.genFull(pset, 1, 2)
# tree = gp.PrimitiveTree(expr)
# f = gp.compile(tree, pset)
# s = runGame(tree)
# print(str(tree))
# print("score:", s)
