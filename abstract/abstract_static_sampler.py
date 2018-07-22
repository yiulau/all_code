import math
import numpy

from general_util.time_diagnostics import time_diagnositcs

def abstract_static_one_step(epsilon, init_q,Ham,evolve_L=None,evolve_t=None,log_obj=None,max_L=500,stepsize_jitter=False):
    # Input:
    # current_q Pytorch Variable
    # H_fun(q,p,return_float) returns Pytorch Variable or float
    # generate_momentum(q) returns pytorch variable
    # Output:
    # accept_rate: float - probability of acceptance
    # accepted: Boolean - True if proposal is accepted, False otherwise
    # divergent: Boolean - True if the end of the trajectory results in a divergent transition
    # return_q  pytorch Variable (! not tensor)
    # q = Variable(current_q.data.clone(),requires_grad=True)
    # evaluate gradient L*2 times
    # evluate H 1 time


    if not evolve_L is None and not evolve_t is None:
        raise ValueError("L contradicts with evol_t")
    assert evolve_L is None or evolve_t is None
    assert not (evolve_L is None and evolve_t is None)
    if stepsize_jitter:
        epsilon = numpy.random.uniform(low=0.9*epsilon,high=1.1*epsilon)
    if not evolve_t is None:
        assert evolve_L is None
        evolve_L = round(evolve_t/epsilon)
        evolve_L = min(evolve_L,max_L)
    careful = True
    Ham.diagnostics = time_diagnositcs()
    divergent = False
    accept_rate = 0
    accepted = False
    explode_grad = False
    num_transitions = evolve_L
    q = init_q.point_clone()
    init_p = Ham.T.generate_momentum(q)
    p = init_p.point_clone()
    #print(q.flattened_tensor)
    #print(p.flattened_tensor)
    Ham_out = Ham.evaluate(q, p)
    current_H = Ham_out["H"]
    current_lp = -Ham_out["V"]
    return_lp = current_lp
    return_H = current_H
    return_q = init_q
    return_p = None
    #print("start q {}".format(init_q.flattened_tensor))
    print("startH {}".format(current_H))


    #newq,newp,stat = Ham.integrator(q, p, epsilon, Ham)
    #print(q.flattened_tensor)
    #print(p.flattened_tensor)
    #newH = Ham.evaluate(newq,newp)
    #print(newH)
    #exit()
    #print(type(evolve_L))
    #exit()
    #print(q.flattened_tensor)
    #print(p.flattened_tensor)
    #print("epsilon is {}".format(epsilon))

    #print("epsilon is {}".format(epsilon))
    for i in range(evolve_L):

        q, p, stat = Ham.integrator(q, p, epsilon, Ham)
        divergent = stat["explode_grad"]
        explode_grad = stat["explode_grad"]
        #print(q.flattened_tensor)
        #print(p.flattened_tensor)
        if not explode_grad:
            Ham_out = Ham.evaluate(q, p)
            temp_H = Ham_out["H"]
            #print("H is {}".format(temp_H))
            if(current_H < temp_H and abs(temp_H-current_H)>1000 or divergent):
                # print("yeye")
                # print(i)
                # print(temp_H)
                # print(current_H)
                # exit()
                return_q = init_q
                return_H = current_H
                return_lp = current_lp
                accept_rate = 0
                accepted = False
                divergent = True
                return_p = None
                num_transitions = i
                break
        else:
            break


    if not divergent and not explode_grad:
        Ham_out = Ham.evaluate(q, p)
        proposed_H = Ham_out["H"]
        proposed_lp = -Ham_out["V"]
        if (current_H < proposed_H and abs(current_H - proposed_H) > 1000):
            return_q = init_q
            return_p = None
            return_H = current_H
            return_lp = current_lp
            accept_rate = 0
            accepted = False


        else:

            accept_rate = math.exp(min(0,current_H - proposed_H))
            if (numpy.random.random(1) < accept_rate):
                accepted = True
                return_q = q
                return_p = p
                return_H = proposed_H
                return_lp =proposed_lp
            else:
                accepted = False
                return_q = init_q
                return_p = init_p
                return_H = current_H
                return_lp = current_lp
    Ham.diagnostics.update_time()
        #print(log_obj is None)
    #endH = Ham.evaluate(q,p)
    #accept_rate = math.exp(min(0, current_H - endH))
    #print("accept_rate {}".format(accept_rate))
    print("accept rate {}".format(accept_rate))
    print("accepted {}".format(accepted))
    print("divergent inside {}".format(divergent))
    print("explode grad {}".format(explode_grad))
    if not divergent and not explode_grad:
        print("endH {}".format(Ham.evaluate(q,p)["H"]))
    #exit()
    if not log_obj is None:
        log_obj.store.update({"prop_H":return_H})
        log_obj.store.update({"log_post":return_lp})
        log_obj.store.update({"accepted":accepted})
        log_obj.store.update({"accept_rate":accept_rate})
        log_obj.store.update({"divergent":divergent})
        log_obj.store.update({"num_transitions":num_transitions})
        log_obj.store.update({"explode_grad":explode_grad})
    #print("second time q {}".format(init_q.flattened_tensor))
    #print("return q {}".format(return_q.flattened_tensor))
    return(return_q,return_p,init_p,return_H,accepted,accept_rate,divergent,num_transitions)


def abstract_static_windowed_one_step(epsilon, init_q, Ham,evolve_L=None,evolve_t=None,careful=True,log_obj=None,max_L=500,stepsize_jitter=None):
    # evaluate gradient 2*L times
    # evluate H function L times

    assert evolve_L is None or evolve_t is None
    if not evolve_L==None and not evolve_t==None:
        raise ValueError("L contradicts with evol_t")

    if not evolve_t is None:
        assert evolve_L is None
        evolve_L = round(evolve_t/epsilon)
        evolve_L = min(max_L,evolve_L)
    Ham.diagnostics = time_diagnositcs()
    divergent = False
    explode_grad = False
    num_transitions = evolve_L
    accepted = False
    accept_rate = 0
    q = init_q
    p_init = Ham.T.generate_momentum(q)
    p = p_init.point_clone()
    Ham_out = Ham.evaluate(q,p)
    logw_prop = -Ham_out["H"]
    current_H = -logw_prop
    current_lp = -Ham_out["V"]
    q_prop = q.point_clone()
    p_prop = p.point_clone()
    q_left,p_left = q.point_clone(),p.point_clone()
    q_right,p_right = q.point_clone(), p.point_clone()

    for i in range(evolve_L):
        o = Ham.windowed_integrator(q_left, p_left,q_right,p_right,epsilon, Ham,logw_prop,q_prop,p_prop)
        divergent = o[7]
        if divergent:
            num_transitions = i
            accept_rate = 0
            accepted = False
            q_prop = init_q
            p_prop = None
            log_wprop = -current_H
            break
        else:
            q_left,p_left,q_right,p_right = o[0:4]
            q_prop, p_prop = o[4], o[5]
            logw_prop = o[6]
            explode_grad = o[8]
            accepted = o[9] or accepted
            accept_rate = o[10]


        #print(o[7])

        #accep_rate_sum += o[5]

    if not divergent:
        return_lp = -Ham.evaluate(q_prop,p_prop)["V"]
    else:
        return_lp = current_lp
    #return(q_prop,accep_rate_sum/L)
    if not log_obj is None:
        log_obj.store.update({"prop_H":-logw_prop})
        log_obj.store.update({"log_post":return_lp})
        log_obj.store.update({"accepted":accepted})
        log_obj.store.update({"accept_rate":accept_rate})
        log_obj.store.update({"divergent":divergent})
        log_obj.store.update({"num_transitions":num_transitions})
        log_obj.store.update({"explode_grad": explode_grad})
    return(q_prop,p_prop,p_init,-logw_prop,accepted,accept_rate,divergent,num_transitions)