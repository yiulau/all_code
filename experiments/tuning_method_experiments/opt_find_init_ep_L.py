# no tuning phase.
import GPyOpt, numpy
class opt_state(object):
    def __init__(self,bounds,init):
        bounds_temp = []
        for i in range(len(bounds)):
            bounds_temp.append({'type': "continuous", 'domain':bounds[i]})
        self.bounds = bounds_temp
        self.store_objective = []
        self.X_step = numpy.array([init])
        self.Y_step = None
    def update(self,new_y):
        if self.Y_step is None:
            self.Y_step = numpy.array([[new_y]])
        else:
            self.Y_step = numpy.vstack((self.Y_step, numpy.array([[new_y]])))
        bo_step = GPyOpt.methods.BayesianOptimization(f=None, domain=self.bounds, X=self.X_step, Y=self.Y_step)
        new_x = bo_step.suggest_next_locations()
        self.X_step = numpy.vstack((self.X_step, new_x))
        return(self.X_step[-1])
    def opt_x(self):
        index = numpy.argmin(self.store_objective)
        best_X = self.X_step[index]
        return(best_X)


# opt_state_obj = opt_state([[0.1,0.5],[1,19]],[0.2,10])
#
# out = opt_state_obj.update(-0.1)
#
# print(out)
#
# out2 = opt_state_obj.update(-0.2)
# print(out2)
#
#



