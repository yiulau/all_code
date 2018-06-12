from abstract.abstract_class_point import point

def gradient_descent(ep,number_of_iter,lr,v_obj):
    dim = v_obj.dim
    # random initialization
    #v_obj.flattened_tensor.normal_()
    #v_obj.load_flattened_to_param()
    init_point = point(V=v_obj)
    init_point.flattened_tensor.normal_()
    init_point.load_flatten()
    theta = init_point.point_clone()
    store_v = []
    for cur in range(number_of_iter):
        theta.flattened_tensor -= lr * v_obj.dq(theta)
        theta.load_flatten()
        store_v.append(v_obj.evaluate_scalar(theta))

        diff = abs(store_v[-1]-store_v[-2])
        if diff < 1e-6:
            break
    return(theta)


