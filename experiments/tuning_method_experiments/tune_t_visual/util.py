from abstract.metric import metric
from abstract.abstract_class_Ham import Hamiltonian
from abstract.abstract_leapfrog_util import abstract_leapfrog_ult

def generate_H_V_T(epsilon,L,vo,q_point):

    metrico = metric("unit_e",vo)
    Ho = Hamiltonian(vo, metrico)
    q = q_point.point_clone()
    p = Ho.T.generate_momentum(q)
    H_0_out = Ho.evaluate(q,p)
    H_list = [H_0_out["H"]]
    V_list = [H_0_out["V"]]
    T_list = [H_0_out["T"]]
    for _ in range(L):
        print("iter {}".format(_))
        q,p,stat = abstract_leapfrog_ult(q, p, epsilon, Ho)
        if stat["explode_grad"]:
            break

        # out = abstract_leapfrog_ult(q,p,epsilon,Ho)
        # out = abstract_NUTS(q,epsilon,Ham,abstract_leapfrog_ult,5)
        H_out = Ho.evaluate(q, p)
        current_H = H_out["H"]
        current_T = H_out["T"]
        current_V = H_out["V"]
        if abs(current_H - H_list[0]) > 1000:
            break
        print("current V is {}".format(current_V))
        print("current H {}".format(current_H))
        print("current T {}".format(current_T))
        # current_V, current_T, current_H = Ho.evaluate_all(q, p)
        # print("current H2 {}".format(Ho.evaluate(q,p)))
        H_list.append(current_H)
        V_list.append(current_V)
        T_list.append(current_T)

    out = {"H_list":H_list,"V_list":V_list,"T_list":T_list}
    return(out)