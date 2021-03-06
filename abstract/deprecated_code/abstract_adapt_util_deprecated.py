import math
import numpy
import torch

from explicit.adapt_util import welford


def find_reasonable_ep(q,p,V,T,integrator):
    # integrator can be leapfrog or gleapfrog using any possible metric
    ep = 1
    H_cur = V(q) + T(p)
    qprime,pprime = integrator(q,p,ep)
    a = 2 * (-V(qprime)-T(pprime) + H_cur > math.log(0.5)) - 1
    while a * (-V(qprime)-T(pprime) + H_cur) > (-a * math.log(2)):
        ep = math.exp(a) * ep
        qprime,pprime = integrator(q,p,ep)
    return(ep)


def dual_averaging_ep(sampler_onestep,generate_momentum,V,T,integrator,q,
                      tune_l=2000, time=1.4, gamma=0.05, t_0=10, kappa=0.75, target_delta=0.65):
    # sampler_onestep should take an q pytorch variable and returns the next accepted q variable as well as acceptance_rate
    # find_reasonable_ep, should only depend on
    # store_ep numpy array storing the epsilons
    p = T.generate_momentum(q)
    ep = find_reasonable_ep(q,p,V,T,integrator)
    mu = math.log(10 * ep)
    bar_ep_i = 1
    bar_H_i = 0

    store_ep = numpy.zeros(tune_l)
    for i in range(tune_l):
        num_step = max(1,round(time/ep))
        out = sampler_onestep(ep, num_step , q, integrator)
        alpha = out[3]
        bar_ep_i, bar_H_i = adapt_ep(alpha,bar_H_i,t_0,i,target_delta,gamma,bar_ep_i,kappa,mu)
        store_ep[i] = bar_ep_i
        ep = bar_ep_i
        q.flattened_tensor = out[0].flattened_tensor
        q.loadfromflatten()
    return(store_ep,q)

def adapt_ep(alpha,bar_H_i,t_0,i,target_delta,gamma,bar_ep_i,kappa,mu):
    bar_H_i = (1 - 1 / (i + 1 + t_0)) * bar_H_i + (1 / (i + 1 + t_0)) * (target_delta - alpha)
    logep = mu - math.sqrt(i + 1) / gamma * bar_H_i
    logbarep = math.pow(i + 1, -kappa) * logep + (1 - math.pow(i + 1, -kappa)) * math.log(bar_ep_i)
    bar_ep_i = math.exp(logbarep)

    return(bar_ep_i,bar_H_i)



def full_adapt(metric,sampler_onestep,generate_momentum,V,T,H,integrator,q,
                      tune_l=2000, time=1.4, gamma=0.05, t_0=10, kappa=0.75, target_delta=0.65):
    # sampler_onestep should take an q pytorch variable and returns the next accepted q variable as well as acceptance_rate
    # find_reasonable_ep, should only depend on
    # store_ep numpy array storing the epsilons
    # covar can be None,dense,or diag
    # adapt both epsilon and covariances
    p = T.generate_momentum(q)
    ep = find_reasonable_ep(q,p,V,T,integrator)
    mu = math.log(10 * ep)
    bar_ep_i = 1
    bar_H_i = 0
    store_ep = numpy.zeros(tune_l)
    if metric=="unit_e":
        for i in range(tune_l):
            num_step = max(1,round(time/ep))
            out = sampler_onestep(ep, num_step , q, integrator, V,T)
            alpha = out[3]
            bar_ep_i, bar_H_i = adapt_ep(alpha,bar_H_i,t_0,i,target_delta,gamma,bar_ep_i,kappa,mu)
            store_ep[i] = bar_ep_i
            ep = bar_ep_i
            q.flattened_tensor = out[0].flattened_tensor
            q.loadfromflatten()
        return(store_ep,q)
    else:
        window_size = 25
        ini_buffer = 75
        end_buffer = 50
        counter_ep = 0
        counter_cov = 0
        dim = len(q)
        update_metric_and_eplist = return_update_metric_ep_list(tune_l,ini_buffer,end_buffer,window_size)
        m_ = torch.zeros(len(p.flattened_tensor))
        if metric.name=="dense_e":
            m_2 = torch.zeros(len(p.flattened_tensor),len(p.flattened_tensor))
        else:
            m_2 = torch.zeros(len(p.flattened_tensor))
        for i in range(tune_l):
            # updates epsilon only in the beginning and at the end
            if i < ini_buffer or i >= tune_l - end_buffer:
                num_step = max(1, round(time / ep))
                out = sampler_onestep(ep, num_step, q, integrator, V,T, generate_momentum)
                alpha = out[3]
                bar_ep_i, bar_H_i = adapt_ep(alpha, bar_H_i, t_0, counter_ep, target_delta, gamma, bar_ep_i, kappa, mu)
                store_ep[i] = bar_ep_i
                ep = bar_ep_i
                counter_ep += 1
                q.flattened_tensor = out[0].flattened_tensor
                q.loadfromflatten()
            else:
                if i in update_metric_and_eplist:
                    # update metric,H function and generate_momentum method. reset counter_cov, accumulators to zero for next window
                    num_step = max(1, round(time / ep))
                    out = sampler_onestep(ep, num_step, q, integrator, H, generate_momentum)
                    alpha = out[3]
                    bar_ep_i, bar_H_i = adapt_ep(alpha, bar_H_i, t_0, counter_ep, target_delta, gamma, bar_ep_i, kappa,
                                                 mu)
                    store_ep[i] = bar_ep_i
                    ep = bar_ep_i
                    counter_ep += 1
                    q.flattened_tensor = out[0].flattened_tensor
                    q.loadfromflatten()
                    m_, m_2, counter_cov = welford(q.flattened_tensor, counter_cov, m_, m_2, metric.name)
                    metric.set_metric(generate_momentum,V,m_2.clone(),metric)
                    if not i == tune_l - end_buffer-1:
                        m_.zero()
                        m_2.zero()
                        counter_cov = 0
                else:
                    m_, m_2, counter_cov = welford(q.flattened_tensor, counter_cov, m_, m_2, metric.name)

    return(store_ep,q,generate_momentum,V,T)

def return_update_metric_ep_list(tune_l,ini_buffer=75,end_buffer=50,window_size=25):
    # returns indices at which the chain ends a covariance update window and also updates epsilon once
    if tune_l < ini_buffer + end_buffer + window_size:
        return("error")
    else:
        cur_window_size = window_size
        counter = ini_buffer
        overshoots = False
        output_list = []
        while not overshoots:
            counter = counter + cur_window_size
            cur_window_size = cur_window_size * 2
            overshoots = counter >= tune_l - end_buffer
            if overshoots:
                output_list.append(tune_l-end_buffer-1)
            else:
                output_list.append(counter-1)
    return(output_list)

def return_update_ep_list(tune_l,ini_buffer=75,end_buffer=50,window_size=25):
    # returns indices at which the chain updates epsilon once
    if tune_l < ini_buffer + end_buffer + window_size:
        return("error")
    else:
        cur_window_size = window_size
        counter = 0
        overshoots = False
        output_list = []
        while not overshoots:
            if counter<ini_buffer:
                counter+=1
                output_list.append(counter - 1)
            else:
                counter = counter + cur_window_size
                cur_window_size = cur_window_size * 2
                overshoots = counter >= tune_l - end_buffer
                if overshoots:
                    output_list.append(tune_l-end_buffer-1)
                else:
                    output_list.append(counter-1)
        if counter >  tune_l - end_buffer:
            output_list.append(counter-1)
    return(output_list)

# def return_update_slow_list(tune_l,ini_buffer=75,end_buffer=50,slow_window_size=25,scale_factor=2):
#     # returns indices at which the chain ends a covariance update window and also updates epsilon once
#     if tune_l < ini_buffer + end_buffer + slow_window_size:
#         raise ValueError("warmup iterations not long enough")
#     else:
#         cur_window_size = slow_window_size
#         counter = self.slow_start
#         overshoots = False
#         output_list = []
#         while not overshoots:
#             counter = counter + cur_window_size
#             cur_window_size = cur_window_size * scale_factor
#             overshoots = (counter >= self.slow_end)
#             #overshoots = counter >= tune_l - end_buffer
#             if overshoots:
#                 output_list.append(tune_l-end_buffer-1)
#             else:
#                 output_list.append(counter-1)
#     return(output_list)


def return_update_lists(tune_l,ini_buffer=75,end_buffer=50,window_size=25,min_medium_updates=10):
    # returns three lists
    # fast list , medium list , slow list
    fast_list = []
    medium_list = []
    slow_list = []
    tune_fast = False
    tune_medium = False
    tune_slow = False
    window_size = 25
    cur_slow_window = window_size

    counter = 0
    if tune_fast==False:
        ini_buffer = 0
        end_buffer = 0
    if tune_medium==False:
        min_medium_updates=0


    first_start_f = 0
    first_end_f = ini_buffer
    second_start_f = tune_l - ini_buffer
    second_end_f = tune_l
    first_start_m = first_end_f
    first_end_m = first_end_f + min_medium_updates*window_size
    second_end_m = second_start_f
    second_start_m = second_end_m - round(min_medium_updates*3/4)*window_size
    start_s = first_end_m
    end_s = second_start_m


    if tune_fast == False:
        first_start_m = first_start_f
        second_end_m = tune_l
    if tune_medium == False:
        start_s = first_start_m
        end_s = second_end_m
    if tune_slow == False:
        first_end_m = second_end_m

    overshoots = False

    counter = 0
    while not overshoots:
        if counter < first_end_f:
            fast_list.append(counter)
            counter += 1
        elif counter < first_end_m and counter >=first_start_m:
            medium_list.append(counter)
            fast_list.append(counter)

            counter += window_size
        elif counter < end_s and counter >= start_s:
            slow_list.append(counter)
            medium_list.append(counter)
            fast_list.append(counter)
            counter += slow_window_size
            slow_window_size = slow_window_size * 2

        elif counter < second_end_m and counter >= second_start_m:
            medium_list.append(counter)
            fast_list.append(counter)
            counter += window_size

        elif counter < second_end_f and counter >= second_start_f:
            fast_list.append(counter)
            counter +=1
        else:
            overshoots = True


    return(fast_list,medium_list,slow_list)












    if not self.tune_fast_parm:
        fast_end = 0
        if not self.tune_medium_param:
            medium_end = 0
            if not self.tune_slow_param:
                slow_end = 0



    if tune_l < ini_buffer + end_buffer + window_size:
        return("error")
    else:
        cur_window_size = window_size
        counter = 0
        overshoots = False
        output_list = []
        while not overshoots:
            if counter<ini_buffer:
                counter+=1
                output_list.append(counter - 1)
            else:
                counter = counter + cur_window_size
                cur_window_size = cur_window_size * 2
                overshoots = counter >= tune_l - end_buffer
                if overshoots:
                    output_list.append(tune_l-end_buffer-1)
                else:
                    output_list.append(counter-1)
        if counter >  tune_l - end_buffer:
            output_list.append(counter-1)
    return(output_list)

def return_update_medium_list(tune_l,ini_buffer=75,end_buffer=50,window_size=25):
    # returns indices at which the chain updates epsilon once
    if tune_l < ini_buffer + end_buffer + window_size:
        return("error")
    else:
        cur_window_size = window_size
        counter = 0
        overshoots = False
        output_list = []
        while not overshoots:
            if counter<ini_buffer:
                counter+=1
                output_list.append(counter - 1)
            else:
                counter = counter + cur_window_size
                cur_window_size = cur_window_size * 2
                overshoots = counter >= tune_l - end_buffer
                if overshoots:
                    output_list.append(tune_l-end_buffer-1)
                else:
                    output_list.append(counter-1)
        if counter >  tune_l - end_buffer:
            output_list.append(counter-1)
    return(output_list)

def return_update_slow_list(tune_l,ini_buffer=75,end_buffer=50,window_size=25):
    # returns indices at which the chain updates epsilon once
    if tune_l < ini_buffer + end_buffer + window_size:
        return("error")
    else:
        cur_window_size = window_size
        counter = 0
        overshoots = False
        output_list = []
        while not overshoots:
            if counter<ini_buffer:
                counter+=1
                output_list.append(counter - 1)
            else:
                counter = counter + cur_window_size
                cur_window_size = cur_window_size * 2
                overshoots = counter >= tune_l - end_buffer
                if overshoots:
                    output_list.append(tune_l-end_buffer-1)
                else:
                    output_list.append(counter-1)
        if counter >  tune_l - end_buffer:
            output_list.append(counter-1)
    return(output_list)