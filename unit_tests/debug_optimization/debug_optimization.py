from abstract.abstract_class_point import point
from input_data.convert_data_to_dict import get_data_dict
from abstract.util import wrap_V_class_with_input_data
from distributions.logistic_regressions.logistic_regression import V_logistic_regression
def gradient_descent(number_of_iter,lr,v_obj):
    # random initialization
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
        #print(theta.flattened_tensor)
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

