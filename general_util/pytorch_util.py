import torch,numpy


def get_list_stats(list_tensor):
    num_var = len(list_tensor)
    store_lens = [None] * num_var
    store_shape = [None] * num_var
    store_slices = [0] * num_var
    dim = 0
    cur = 0
    for i in range(num_var):
        store_shape[i] = list(list_tensor[i].shape)
        store_lens[i] = int(numpy.prod(store_shape[i]))
        dim += store_lens[i]
        store_slices[i] = numpy.s_[cur:(cur+store_lens[i])]
        cur = cur+store_lens[i]
    return (store_shape, store_lens, dim,store_slices)



class welford(object):
    def __init__(self):
        self.iter = 0
        self.m_ = 0
        self.m_2 = 0

    def mean(self,next_sample,m_):
        delta = (next_sample - m_)
        m_ += delta / self.iter
        self.iter += 1
        self.m_ = m_
        self.m_2  += (next_sample-m_) * delta
        return(m_)


def general_load_point(obj,point_obj):
    # obj = V_obj or T_obj
        #print("point syncro {}".format(q_point.assert_syncro()))
    if obj.need_flatten:
        obj.flattened_tensor.copy_(point_obj.flattened_tensor)
        obj.load_param_to_flattened()
    else:
        #self.flattened_tensor.copy_(q_point.flattened_tensor)
        for i in range(obj.num_var):
            obj.list_tensor[i].copy_(point_obj.list_tensor[i])
    #print("q_point syncro {}".format(q_point.assert_syncro()))
    #print("self syncro {}".format(self.assert_syncro()))
    return()

def general_assert_syncro(obj):
    # obj = V_obj or T_obj
    # check that list_tensor and flattened_tensor hold the same value
    temp = obj.flattened_tensor.clone()
    cur = 0
    for i in range(obj.num_var):
        temp[cur:(cur + obj.store_lens[i])].copy_(obj.list_tensor[i].view(obj.store_shapes[i]))
        cur = cur + obj.store_lens[i]
    diff = ((temp - obj.flattened_tensor)*(temp-obj.flattened_tensor) ).sum()
    #print(diff)
    if diff > 1e-6:
        out = False
    else:
        out = True
    return(out)

def general_load_param_to_flattened(obj):
    cur = 0
    for i in range(obj.num_var):
        obj.flattened_tensor[cur:(cur + obj.store_lens[i])].copy_(obj.list_tensor[i].view(obj.store_shapes[i]))
        cur = cur + obj.store_lens[i]
    return()

def general_load_flattened_tensor_to_param(obj,flattened_tensor=None):
    cur = 0
    if flattened_tensor is None:
        for i in range(obj.num_var):
            # convert to copy_ later
            obj.list_tensor[i].copy_(obj.flattened_tensor[cur:(cur + obj.store_lens[i])].view(obj.store_shapes[i]))
    else:
        for i in range(obj.num_var):
            # convert to copy_ later
            obj.list_tensor[i].copy_(flattened_tensor[cur:(cur + obj.store_lens[i])].view(obj.store_shapes[i]))
        obj.load_param_to_flattened()
    return()