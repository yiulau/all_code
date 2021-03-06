import math
import numpy
import torch
from explicit.general_util import logsumexp, stable_sum

from general_util.time_diagnostics import time_diagnositcs


def abstract_NUTS(init_q,epsilon,Ham,max_tree_depth=5,log_obj=None):
    # input and output are point objects
    Ham.diagnostics = time_diagnositcs()
    p_init = Ham.T.generate_momentum(init_q)

    q_left = init_q.point_clone()
    q_right =init_q.point_clone()
    p_left = p_init.point_clone()
    p_right = p_init.point_clone()
    j = 0
    num_div = 0
    q_prop = init_q.point_clone()
    p_prop = p_init.point_clone()
    Ham_out = Ham.evaluate(init_q, p_init)
    log_w = -Ham_out["H"]
    H_0 = -log_w
    lp_0 = -Ham_out["V"]
    accepted = False
    accept_rate = 0
    divergent = False
    s = True
    diagn_dict = {"divergent":None,"explode_grad":None}
    while s:
        v = numpy.random.choice([-1,1])
        if v < 0:
            q_left, p_left, _, _, q_prime, p_prime,s_prime, log_w_prime,num_div_prime = abstract_BuildTree_nuts(q_left, p_left, -1, j, epsilon, Ham,H_0,diagn_dict
                                                                            )
        else:
            _, _, q_right, p_right, q_prime, p_prime, s_prime, log_w_prime,num_div_prime = abstract_BuildTree_nuts(q_right, p_right, 1, j, epsilon, Ham,H_0,diagn_dict
                                                                              )

        if s_prime:
            accept_rate = math.exp(min(0, (log_w_prime - log_w)))
            u = numpy.random.rand(1)
            if u < accept_rate:
                accepted = accepted or True
                q_prop = q_prime.point_clone()
                p_prop = p_prime.point_clone()

            log_w = logsumexp(log_w,log_w_prime)
            s = s_prime and abstract_NUTS_criterion(q_left,q_right,p_left,p_right)
            j = j + 1
            s = s and (j<max_tree_depth)
        else:
            s = False
        num_div += num_div_prime
        Ham.diagnostics.update_time()
    if num_div >0:
        divergent = True
        p_prop = None
        return_lp = lp_0
    else:
        return_lp = -Ham.evaluate(q_prop,p_prop)["V"]



    if not log_obj is None:
        log_obj.store.update({"prop_H":-log_w})
        log_obj.store.update({"log_post":return_lp})
        log_obj.store.update({"accepted":accepted})
        log_obj.store.update({"accept_rate":accept_rate})
        log_obj.store.update({"divergent":divergent})
        log_obj.store.update({"tree_depth":j})
        log_obj.store.update({"num_transitions":math.pow(2,j)})
        log_obj.store.update({"hit_max_tree_depth":j>=max_tree_depth})
    return(q_prop,p_prop,p_init,-log_w,accepted,accept_rate,divergent,j)
def abstract_GNUTS(init_q,epsilon,Ham,max_tree_depth=5,log_obj=None):
    # sum_p should be a tensor instead of variable

    Ham.diagnostics = time_diagnositcs()
    p_init = Ham.T.generate_momentum(init_q)
    q_left = init_q.point_clone()
    q_right = init_q.point_clone()
    p_left = p_init.point_clone()
    p_right = p_init.point_clone()
    p_sleft = Ham.p_sharp_fun(init_q, p_init).point_clone()
    p_sright = Ham.p_sharp_fun(init_q, p_init).point_clone()
    j = 0
    num_div = 0
    q_prop = init_q.point_clone()
    p_prop = p_init.point_clone()
    Ham_out = Ham.evaluate(init_q,p_init)
    log_w = -Ham_out["H"]
    H_0 = -log_w
    lp_0 = -Ham_out["V"]
    accepted = False
    accept_rate = 0
    divergent = False
    sum_p = p_init.flattened_tensor.clone()
    s = True
    diagn_dict = {"divergent":None,"explode_grad":None}
    while s:
        v = numpy.random.choice([-1,1])
        if v < 0:
            q_left, p_left, _, _, q_prime, p_prime,s_prime, log_w_prime,sum_dp,num_div_prime = abstract_BuildTree_gnuts(q_left, p_left, -1, j, epsilon, Ham,
                                                                            H_0,diagn_dict)
        else:
            _, _, q_right, p_right, q_prime, p_prime,s_prime, log_w_prime, sum_dp,num_div_prime = abstract_BuildTree_gnuts(q_right, p_right, 1, j, epsilon, Ham,
                                                                              H_0,diagn_dict)

        if s_prime:
            accept_rate = math.exp(min(0, (log_w_prime - log_w)))
            u = numpy.random.rand(1)
            if u < accept_rate:
                accepted = accepted or True
                q_prop = q_prime.point_clone()
                p_prop = p_prime.point_clone()
            log_w = logsumexp(log_w,log_w_prime)
            sum_p += sum_dp
            p_sleft = Ham.p_sharp_fun(q_left, p_left)
            p_sright = Ham.p_sharp_fun(q_right, p_right)
            s = s_prime and abstract_gen_NUTS_criterion(p_sleft,p_sright,sum_p)
            j = j + 1
            s = s and (j < max_tree_depth)
        else:
            s = False
            #accept_rate = 0
            #divergent = True


        num_div +=num_div_prime
    Ham.diagnostics.update_time()
    if num_div > 0 :
        divergent = True
        p_prop = None
        return_lp = lp_0
    else:
        return_lp = -Ham.evaluate(q_prop,p_prop)["V"]

    if not log_obj is None:
        log_obj.store.update({"prop_H":-log_w})
        log_obj.store.update({"log_post":return_lp})
        log_obj.store.update({"accepted":accepted})
        log_obj.store.update({"accept_rate":accept_rate})
        log_obj.store.update({"divergent":divergent})
        log_obj.store.update({"explode_grad":diagn_dict["explode_grad"]})
        log_obj.store.update({"tree_depth":j})
        log_obj.store.update({"num_transitions":math.pow(2,j)})
        log_obj.store.update({"hit_max_tree_depth":j>=max_tree_depth})
    return(q_prop,p_prop,p_init,-log_w,accepted,accept_rate,divergent,j)
def abstract_NUTS_xhmc(init_q,epsilon,Ham,xhmc_delta,max_tree_depth=5,log_obj=None,debug_dict=None):
    Ham.diagnostics = time_diagnositcs()
    #seedid = 30
    #numpy.random.seed(seedid)
    #torch.manual_seed(seedid)
    p_init = Ham.T.generate_momentum(init_q)
    q_left = init_q.point_clone()
    q_right = init_q.point_clone()
    p_left = p_init.point_clone()
    p_right = p_init.point_clone()
    j = 0
    num_div = 0
    q_prop = init_q.point_clone()
    p_prop = p_init.point_clone()
    Ham_out = Ham.evaluate(init_q, p_init)
    log_w = -Ham_out["H"]
    H_0 = -log_w
    lp_0 = -Ham_out["V"]
    accepted = False
    accept_rate = 0
    divergent = False
    ave = Ham.dG_dt(init_q, p_init)
    diagn_dict = {"divergent":None,"explode_grad":None}
    s = True
    while s:
        v = numpy.random.choice([-1,1])
        #print("j {}".format(j==6))
        #print("abstract v {}".format(v))
        if v < 0:
            q_left, p_left, _, _, q_prime,p_prime, s_prime, log_w_prime,ave_dp,num_div_prime = abstract_BuildTree_nuts_xhmc(q_left, p_left, -1, j, epsilon, Ham,
                                                                            xhmc_delta,H_0,diagn_dict)
        else:


            _, _, q_right, p_right, q_prime,p_prime, s_prime, log_w_prime,ave_dp,num_div_prime = abstract_BuildTree_nuts_xhmc(q_right, p_right, 1, j, epsilon, Ham,
                                                                              xhmc_delta,H_0,diagn_dict)
        # if j == 2:
        #     print("abstract q_prime {}".format(q_prime.flattened_tensor))
        #
        #
        # if j ==1:
        #     #print("abstract ar {}".format(accept_rate))
        #     pass
        #     #print("abstract pprime {}".format(p_prime.flattened_tensor))
        # #print("abstract s_prime {}".format(s_prime))
        if s_prime:
            accept_rate = math.exp(min(0, (log_w_prime - log_w)))
            u = numpy.random.rand(1)
            if u < accept_rate:
                accepted = accepted or True
                q_prop = q_prime.point_clone()
                p_prop = p_prime.point_clone()
            oo = stable_sum(ave, log_w, ave_dp, log_w_prime)
            ave = oo[0]
            log_w = oo[1]
            s = s_prime and abstract_xhmc_criterion(ave,xhmc_delta,math.pow(2,j))
            j = j + 1
            s = s and (j < max_tree_depth)
        else:
            s = False

        num_div +=num_div_prime
    Ham.diagnostics.update_time()
    if num_div > 0:
        divergent = True
        p_prop = None
        return_lp = lp_0
    else:
        return_lp = -Ham.evaluate(q_prop,p_prop)["V"]

    if not log_obj is None:
        log_obj.store.update({"prop_H":-log_w})
        log_obj.store.update({"log_post":return_lp})
        log_obj.store.update({"accepted":accepted})
        log_obj.store.update({"accept_rate":accept_rate})
        log_obj.store.update({"divergent":divergent})
        log_obj.store.update({"explode_grad":diagn_dict["explode_grad"]})
        log_obj.store.update({"tree_depth":j})
        log_obj.store.update({"num_transitions":math.pow(2,j)})
        log_obj.store.update({"hit_max_tree_depth": j >= max_tree_depth})
    #print("abstract num_div {}".format(num_div))
    #debug_dict.update({"abstract": j})

    return(q_prop,p_prop,p_init,-log_w,accepted,accept_rate,divergent,j)
def abstract_BuildTree_nuts(q,p,v,j,epsilon,Ham,H_0,diagn_dict):
    if j ==0:
        q_prime,p_prime,stat = Ham.integrator(q,p,v*epsilon,Ham)

        divergent = stat["explode_grad"]
        diagn_dict.update({"explode_grad":divergent})
        diagn_dict.update({"divergent":divergent})
        if not divergent :
            # continue_divergence
            # boolean True if there's no divergence.
            log_w_prime = -Ham.evaluate(q_prime, p_prime)["H"]
            H_cur = -log_w_prime
            if abs(H_cur-H_0)<1000:
                continue_divergence = True
                num_div = 0
            else:
                diagn_dict.update({"divergent":divergent})
                continue_divergnce = False
                num_div = 1
                raise ValueError("definitely divergent")
        else:
            continue_divergence = False
            num_div = 1
        return q_prime.point_clone(), p_prime.point_clone(), q_prime.point_clone(), p_prime.point_clone(), q_prime.point_clone(), p_prime.point_clone(), continue_divergence, log_w_prime, num_div
    else:
        # first half of subtree
        q_left, p_left, q_right, p_right, q_prime, p_prime,s_prime, log_w_prime,num_div_prime = abstract_BuildTree_nuts(q, p, v, j - 1, epsilon, Ham,H_0,diagn_dict)
        # second half of subtree
        if s_prime:
            if v <0:
                q_left,p_left,_,_,q_dprime,p_dprime,s_dprime,log_w_dprime,num_div_dprime = abstract_BuildTree_nuts(q_left,p_left,v,j-1,epsilon,Ham,H_0,diagn_dict)
            else:
                _, _, q_right, p_right, q_dprime,p_dprime, s_dprime, log_w_dprime,num_div_dprime = abstract_BuildTree_nuts(q_right, p_right, v, j - 1, epsilon, Ham,H_0,diagn_dict)

            if s_dprime:
                accept_rate = math.exp(min(0,(log_w_dprime-logsumexp(log_w_prime,log_w_dprime))))
                u = numpy.random.rand(1)[0]
                if u < accept_rate:
                    q_prime = q_dprime.point_clone()
                    p_prime = p_dprime.point_clone()
                s_prime = s_dprime and abstract_NUTS_criterion(q_left,q_right,p_left,p_right)
                num_div_prime +=num_div_dprime
                log_w_prime = logsumexp(log_w_prime,log_w_dprime)
        return q_left, p_left, q_right, p_right, q_prime, p_prime, s_prime, log_w_prime,num_div_prime

def abstract_BuildTree_gnuts(q,p,v,j,epsilon,Ham,H_0,diagn_dict):

    #p_sharp_fun(q,p) takes tensor returns tensor

    if j ==0:
        q_prime,p_prime,stat = Ham.integrator(q,p,v*epsilon,Ham)
        divergent = stat["explode_grad"]
        diagn_dict.update({"explode_grad":divergent})
        diagn_dict.update({"divergent":divergent})
        if not divergent:
            log_w_prime = -Ham.evaluate(q_prime, p_prime)["H"]
            H_cur = -log_w_prime
            if (abs(H_cur - H_0) < 1000):
                continue_divergence = True
                num_div = 0
            else:
                diagn_dict.update({"divergent":True})
                continue_divergence = False
                num_div = 1
        else:
            log_w_prime = None
            continue_divergence = False
            num_div = 1

        if not continue_divergence:
            return None, None, None, None, None,None, continue_divergence, log_w_prime, None, num_div
        else:
            return q_prime.point_clone(), p_prime.point_clone(), q_prime.point_clone(), p_prime.point_clone(), q_prime.point_clone(),p_prime.point_clone(), continue_divergence, log_w_prime,p_prime.flattened_tensor.clone(),num_div
    else:
        # first half of subtree
        sum_p = torch.zeros(len(p.flattened_tensor))
        q_left, p_left, q_right, p_right, q_prime,p_prime, s_prime, log_w_prime,temp_sum_p,num_div_prime = abstract_BuildTree_gnuts(q, p, v, j - 1,
                                                                                                     epsilon,
                                                                                                     Ham,H_0,diagn_dict)

            # second half of subtree
        if s_prime:
            sum_p += temp_sum_p
            if v <0:
                q_left,p_left,_,_,q_dprime,p_dprime,s_dprime,log_w_dprime,sum_dp,num_div_dprime = abstract_BuildTree_gnuts(q_left,p_left,v,j-1,epsilon,
                                                                                          Ham,
                                                                                          H_0,diagn_dict)
            else:
                _, _, q_right, p_right, q_dprime, p_dprime,s_dprime, log_w_dprime,sum_dp,num_div_dprime = abstract_BuildTree_gnuts(q_right, p_right, v, j - 1, epsilon,
                                                                                 Ham,H_0,diagn_dict)
            if s_dprime:
                accept_rate = math.exp(min(0,(log_w_dprime-logsumexp(log_w_prime,log_w_dprime))))
                u = numpy.random.rand(1)[0]
                if u < accept_rate:
                    q_prime = q_dprime.point_clone()
                    p_prime = p_dprime.point_clone()
                sum_p += sum_dp
                num_div_prime +=num_div_dprime
                p_sleft = Ham.p_sharp_fun(q_left,p_left)
                p_sright = Ham.p_sharp_fun(q_right,p_right)
                s_prime = s_dprime and abstract_gen_NUTS_criterion(p_sleft, p_sright, sum_p)
                log_w_prime = logsumexp(log_w_prime,log_w_dprime)
            else:
                s_prime = s_dprime and s_prime

        return q_left, p_left, q_right, p_right, q_prime,p_prime, s_prime, log_w_prime,sum_p,num_div_prime
def abstract_BuildTree_nuts_xhmc(q,p,v,j,epsilon,Ham,xhmc_delta,H_0,diagn_dict):
    if j ==0:
        q_prime,p_prime,stat = Ham.integrator(q,p,v*epsilon,Ham)
        divergent = stat["explode_grad"]
        diagn_dict.update({"explode_grad": divergent})
        diagn_dict.update({"divergent": divergent})
        if not divergent:
            log_w_prime = -Ham.evaluate(q_prime, p_prime)["H"]
            H_cur = -log_w_prime
            if(abs(H_cur-H_0)<1000):
                continue_divergence = True
                num_div = 0
                ave= Ham.dG_dt(q_prime,p_prime)
            else:
                diagn_dict.update({"divergent":divergent})
                continue_divergence = False
                num_div = 1
                ave = None
                log_w_prime = None

        else:
            continue_divergence = False
            num_div = 1
            ave = None
            log_w_prime = None
        if not continue_divergence:
            return None,None,None,None,None,None,continue_divergence,log_w_prime,ave,num_div
        else:
            return q_prime.point_clone(), p_prime.point_clone(), q_prime.point_clone(), p_prime.point_clone(), q_prime.point_clone(), p_prime.point_clone(), continue_divergence, log_w_prime, ave, num_div

    else:
        # first half of subtree
        q_left, p_left, q_right, p_right, q_prime,p_prime, s_prime, log_w_prime, ave_prime,num_div_prime = abstract_BuildTree_nuts_xhmc(q, p, v, j - 1,
                                                                                                         epsilon,
                                                                                                         Ham,xhmc_delta,H_0,diagn_dict)
        # second half of subtree
        if s_prime:
            if v <0:
                q_left,p_left,_,_,q_dprime,p_dprime,s_dprime,log_w_dprime, ave_dprime,num_div_dprime = abstract_BuildTree_nuts_xhmc(q_left,p_left,v,j-1,
                                                                                              epsilon,
                                                                                              Ham,xhmc_delta,H_0,diagn_dict)
            else:
                _, _, q_right, p_right, q_dprime,p_dprime, s_dprime, log_w_dprime, ave_dprime,num_div_dprime = abstract_BuildTree_nuts_xhmc(q_right, p_right, v, j - 1, epsilon,
                                                                                 Ham,xhmc_delta,H_0,diagn_dict)
            if s_dprime:
                accept_rate = math.exp(min(0,(log_w_dprime-logsumexp(log_w_prime,log_w_dprime))))
                u = numpy.random.rand(1)[0]
                if u < accept_rate:
                    q_prime = q_dprime.point_clone()
                    p_prime = p_dprime.point_clone()
                oo_ = stable_sum(ave_prime, log_w_prime, ave_dprime, log_w_prime)
                ave_prime = oo_[0]
                log_w_prime = oo_[1]
                num_div_prime += num_div_dprime
                s_prime = s_dprime and abstract_xhmc_criterion(ave_prime,xhmc_delta,math.pow(2,j))
            else:
                s_prime = s_dprime and s_prime

        return q_left, p_left, q_right, p_right, q_prime, p_prime,s_prime,log_w_prime,ave_prime,num_div_prime

def abstract_NUTS_criterion(q_left,q_right,p_left,p_right):
    # True = continue doubling the trajectory
    # False = stop
    o = (torch.dot(p_right.flattened_tensor,q_right.flattened_tensor-q_left.flattened_tensor) >=0) and \
        (torch.dot(p_left.flattened_tensor,q_right.flattened_tensor-q_left.flattened_tensor) >=0)
    return(o)

def abstract_gen_NUTS_criterion(p_sleft,p_sright,p_sum):
    # p_sum should be a tensor
    # True = continue doubling the trajectory
    # False = stop
    o = (torch.dot(p_sleft.flattened_tensor,p_sum) > 0) and \
        (torch.dot(p_sright.flattened_tensor,p_sum) > 0)
    return(o)

def abstract_xhmc_criterion(ave,xhmc_delta,traj_len):
    o = abs(ave)/traj_len > xhmc_delta
    #o = (abs(ave)>xhmc_delta)
    return(o)
