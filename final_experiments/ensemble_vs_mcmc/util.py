from abstract.abstract_class_point import point
from input_data.convert_data_to_dict import get_data_dict
from abstract.util import wrap_V_class_with_input_data
from post_processing.test_error import test_error
import numpy

def gradient_descent(number_of_iter,lr,v_obj,validation_set,validate_interval=10):
    # random initialization
    init_point = point(V=v_obj)
    init_point.flattened_tensor.normal_()
    init_point.load_flatten()
    theta = init_point.point_clone()
    store_v = []
    best_validate_error = 10
    explode_grad = False
    till_validate = validate_interval
    validate_continue = True
    for cur in range(number_of_iter):
        print("iter {}".format(cur))
        if not validate_continue:
            break
        else:
            #print(cur)
            cur_v = v_obj.evaluate_scalar(theta)
            print("v val {}".format(cur_v))
            #print(theta.flattened_tensor)
            grad,explode_grad = v_obj.dq(theta.flattened_tensor)
            if not explode_grad:
                theta.flattened_tensor -= lr * grad
                theta.load_flatten()
                store_v.append(v_obj.evaluate_scalar(theta))
                if till_validate==0:
                    temp_mcmc_samples = numpy.zeros((1,len(theta.flattened_tensor)))
                    temp_mcmc_samples[0,:] = theta.flattened_tensor.numpy()
                    validate_error,_,_ = test_error(target_dataset=validation_set,v_obj=v_obj,mcmc_samples=temp_mcmc_samples,type="classification")
                    print("validate error {}".format(validate_error))
                    if validate_error > best_validate_error:
                        validate_continue = False
                    else:
                        till_validate = validate_interval
                        best_validate_error = validate_error
                else:
                    till_validate-=1
                diff = abs(store_v[-1]-cur_v)
                if diff < 1e-6:
                    break
            else:
                break
    return(theta,explode_grad)

