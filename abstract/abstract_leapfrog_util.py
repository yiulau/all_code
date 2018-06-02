import math
import numpy

from general_util.time_diagnostics import time_diagnositcs


# all functions modify (q,p)

class gleapfrog_stat(object):
    #collects information from start to beginning of one leapfrog/genleapfrog step
    def __init__(self):
        self.divergent = False
        self.first_divergent = None
        self.second_divergent = None

def abstract_leapfrog_ult(q,p,epsilon,Ham):
    # Input:
    # q, p point objects
    # epsilon float
    # H_fun(q,p,return_float) function that maps (q,p) to its energy . Should return a pytorch Variable
    # in this implementation the original (q,p) is modified after running leapfrog(q,p)
    # evaluate gradient 2 times
    # evaluate H 0 times
    #print("yes")
    #print(Ham.evaluate(q,p))
    #out = {"q_tensor":q.flattened_tensor.clone(),"p_tensor":p.flattened_tensor.clone()}
    #import pickle
    #with open('debugqp.pkl', 'wb') as f:
    #    pickle.dump(out, f)
    #exit()

    p.flattened_tensor -= Ham.V.dq(q.flattened_tensor) * 0.5 * epsilon
    #print("first p abstract{}".format(p.flattened_tensor))
    #print("first H abstract {}".format(Ham.evaluate(q,p)))
    q.flattened_tensor += Ham.T.dp(p.flattened_tensor) * epsilon
    #print("first q abstract {}".format(q.flattened_tensor))
    #print("second H abstract {}".format(Ham.evaluate(q,p)))
    p.flattened_tensor -= Ham.V.dq(q.flattened_tensor) * 0.5 * epsilon
    #print("second p abstract {}".format(p.flattened_tensor))
    #print("final q abstract {}".format(q.flattened_tensor))
    p.load_flatten()
    q.load_flatten()
        #print(Ham.evaluate(q,p))


    #exit()
    return(q,p,gleapfrog_stat())






def windowerize(integrator):
    def windowed_integrator(q_left, p_left, q_right, p_right, epsilon, Ham, logw_old, qprop_old, pprop_old):
        # Input: q,p current (q,p) state in trajecory
        # q,p point objects
        # qprop_old, pprop_old current proposed states in trajectory
        # logw_old = -H(qprop_old,pprop_old,return_float=True)
        # evaluate gradient 2 times
        # evaluate H 1 time

        v = numpy.random.choice([-1, 1])

        if v < 0:
            q_left, p_left,stat = integrator(q_left, p_left, v * epsilon, Ham)
            divergent = stat.divergent
            logw_prop = -Ham.evaluate(q_left, p_left)

        else:
            q_right, p_right,stat = integrator(q_right, p_right, v * epsilon, Ham)
            divergent = stat.divergent
            logw_prop = -Ham.evaluate(q_right, p_right)

        if (abs(logw_prop - logw_old) > 1000 or divergent):
            accept_rate = 0
            divergent = True

        else:
            # uniform progressive sampling
            # accept_rate = math.exp(min(0, (logw_prop - logsumexp(logw_prop, logw_old))))
            # baised progressive sampling
            accept_rate = math.exp(min(0, logw_prop - logw_old))
            divergent = False
        u = numpy.random.rand(1)[0]
        if u < accept_rate:
            qprop = q_right
            pprop = p_right
            accepted = True
        else:
            qprop = qprop_old
            pprop = pprop_old
            logw_prop = logw_old
            accepted = False

        return (q_left, p_left, q_right, p_right, qprop, pprop, logw_prop, divergent, accepted, accept_rate)
    return(windowed_integrator)