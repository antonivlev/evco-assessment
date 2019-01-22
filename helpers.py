from deap import gp
from deap import base
from deap import creator
from deap import tools

def toolboxConfiguration(snake=[], pset=[], eval_func=lambda: 0):
	#followPath()
	#pset.addPrimitive(snake.if_food_ahead, 2)
	# pset.addPrimitive(snake.if_wall_ahead, 2)
	# pset.addPrimitive(snake.if_tail_ahead, 2)
	# pset.addPrimitive(snake.if_death_left, 2)
	# pset.addPrimitive(snake.if_death_right, 2)
	# pset.addPrimitive(snake.if_death_up, 2)
	# pset.addPrimitive(snake.if_death_down, 2)
	# pset.addPrimitive(snake.if_moving_up, 2)
	# pset.addPrimitive(snake.if_moving_down, 2)
	# pset.addPrimitive(snake.if_moving_left, 2)
	# pset.addPrimitive(snake.if_moving_right, 2)
	# pset.addPrimitive(snake.if_right_gets_closer_to_food, 2)
	# pset.addPrimitive(snake.if_left_gets_closer_to_food, 2)
	# pset.addPrimitive(snake.if_up_gets_closer_to_food, 2)
	# pset.addPrimitive(snake.if_down_gets_closer_to_food, 2)

	pset.addTerminal(snake.changeDirectionUp)
	pset.addTerminal(snake.changeDirectionDown)
	pset.addTerminal(snake.changeDirectionLeft)
	pset.addTerminal(snake.changeDirectionRight)

	creator.create("FitnessFunc", base.Fitness, weights=(1.0,1.0,))
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
	toolbox.register("evaluate", eval_func)

	return toolbox
