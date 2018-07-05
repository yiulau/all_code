# no tuning phase.
import GPyOpt, numpy
# bounds = [epsilon,t]
class opt_state(object):
    def __init__(self,bounds,init,L_upper_bound=None):
        bounds_temp = []
        for i in range(len(bounds)):
            bounds_temp.append({'type': "continuous", 'domain':bounds[i]})
        self.bounds = bounds_temp
        self.store_objective = []
        self.X_step = numpy.array([init])
        self.Y_step = None
        self.L_upper_bound = L_upper_bound
    def update(self,new_y):
        if self.Y_step is None:
            self.Y_step = numpy.array([[new_y]])
        else:
            self.Y_step = numpy.vstack((self.Y_step, numpy.array([[new_y]])))
        if not self.L_upper_bound is None:
            assert self.L_upper_bound >0
            constraints = [{"name": "const_1", "constraint": " x[:, 1]/x[:, 0] - {}".format(self.L_upper_bound)}]
            bo_step = GPyOpt.methods.BayesianOptimization(f=None, domain=self.bounds, X=self.X_step, Y=self.Y_step,constraints=constraints)
        else:
            bo_step = GPyOpt.methods.BayesianOptimization(f=None, domain=self.bounds, X=self.X_step, Y=self.Y_step)
        new_x = bo_step.suggest_next_locations()
        self.X_step = numpy.vstack((self.X_step, new_x))
        return(self.X_step[-1])
    def opt_x(self):
        index = numpy.argmin(self.store_objective)
        best_X = self.X_step[index]
        return(best_X)


#opt_state_obj = opt_state([[0.1,0.5],[0.7,15]],[0.2,10],L_upper_bound=100)
#
#out = opt_state_obj.update(-0.1)
#
#print(out)
#
#out2 = opt_state_obj.update(-0.2)
#print(out2)
#
#
#out2 = opt_state_obj.update(-0.05)
#print(out2)



