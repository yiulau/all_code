from abstract.abstract_class_point import point
from distributions.logistic_regressions.pima_indian_logisitic_regression import V_pima_inidan_logit
def gradient_descent(number_of_iter,lr,v_obj):
    dim = v_obj.dim
    # random initialization
    #v_obj.flattened_tensor.normal_()
    #v_obj.load_flattened_to_param()
    init_point = point(V=v_obj)
    init_point.flattened_tensor.normal_()
    init_point.load_flatten()
    theta = init_point.point_clone()
    store_v = []
    explode_grad = False
    for cur in range(number_of_iter):
        print(cur)
        cur_v = v_obj.evaluate_scalar(theta)
        print(cur_v)
        print(theta.flattened_tensor)
        grad,explode_grad = v_obj.dq(theta.flattened_tensor)
        if not explode_grad:
            theta.flattened_tensor -= lr * grad
            theta.load_flatten()
            store_v.append(v_obj.evaluate_scalar(theta))

            diff = abs(store_v[-1]-cur_v)
            if diff < 1e-6:
                break
        else:
            break
    return(theta,explode_grad)

v_obj = V_pima_inidan_logit()

out,explode_grad = gradient_descent(number_of_iter=100,lr=0.01,v_obj=v_obj)






