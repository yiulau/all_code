import GPyOpt, numpy
class opt_state(object):
    def __init__(self,bounds,init):
        self.bounds = bounds
        self.store_objective = []
        self.X_step = []
        self.Y_step = []
    def update(self,new_y):
        self.Y_step = numpy.array([[new_y]])
        bo_step = GPyOpt.methods.BayesianOptimization(f=None, domain=self.bounds, X=self.X_step, Y=self.Y_step)
        new_x = bo_step.suggest_next_locations()
        self.X_step = numpy.vstack((self.X_step, new_x))
        return(new_x)
    def opt_x(self):
        index = numpy.argmin(self.store_objective)
        best_X = self.X_step[index]
        return(best_X)



