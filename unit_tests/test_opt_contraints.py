import GPyOpt
import GPy
import numpy as np


space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0.001,0.1)},
        {'name': 'var_2', 'type': 'continuous', 'domain': (1.5,55)}]
constraints = [{'name': 'constr_1', 'constrain': 'x[:,1]/x[:,0]-150'}]

feasible_region = GPyOpt.Design_space(space = space, constraints = constraints)

initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, 1)

print(initial_design)
print(initial_design[:,1]/initial_design[:,0])
y_step = np.array([[-2.1]])
def out(x):
    return(y_step)

# --- CHOOSE the objective
objective = out

# --- CHOOSE the model type
model = GPyOpt.models.GPModel(exact_feval=True,optimize_restarts=10,verbose=False)

# --- CHOOSE the acquisition optimizer
aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)

# --- CHOOSE the type of acquisition
acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)

# --- CHOOSE a collection method
evaluator = GPyOpt.core.evaluators.Sequential(acquisition)


bo = GPyOpt.methods.BayesianOptimization(model, feasible_region, objective, acquisition, evaluator,initial_design)

out = bo.suggest_next_locations()
