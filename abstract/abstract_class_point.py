from torch.autograd import Variable
import numpy, torch
# how a point is defined depends on the problem at hand. if nn then we might not want an extra copy of the weights
class point(object):
    # numpy vector of pytorch variables
    # need to be able to calculate  norm(q-p)_2 (esjd)
    # sum two points
    # clone a point
    # multiply by scalar
    # sum load (target, summant)
    # return list of tensors
    # dot product
    # need_flatten depends on metric. anything other than unit_e and diag_e need_flatten = True
    # point does not /should not need to contain variables.
    def __init__(self,V=None,T=None,list_var=None,list_tensor=None,pointtype=None,
                 need_flatten=True):
        # self.V = V
        # self.T = T
        #self.list_var= list_var
        #self.list_tensor = list_tensor
        self.pointtype = pointtype
        self.need_flatten = need_flatten
        if not V is None:
            self.pointtype = "q"
            target_fun_obj = V
            self.need_flatten = target_fun_obj.need_flatten
            self.num_var = target_fun_obj.num_var
            source_list_tensor = target_fun_obj.list_tensor
        elif not T is None:
            self.pointtype = "p"
            target_fun_obj = T
            self.need_flatten = target_fun_obj.need_flatten
            self.num_var = target_fun_obj.num_var
            source_list_tensor = target_fun_obj.list_tensor
        else:
            if list_var is None and list_tensor is None:
                raise ValueError("one of V,T or a list of variables, or a list of tensors must be supplied")
            else:
                if not list_var is None and not list_tensor is None:
                    raise ValueError("can't supply both list_tensor and list_var. need exactly one ")
                assert pointtype in ("p","q")
                self.pointtype = pointtype

                assert not list_tensor is None
                self.num_var = len(list_tensor)
                if list_tensor is None:
                    source_list_tensor = []
                    assert not list_var is None
                    for i in range(len(list_var)):
                        source_list_tensor.append(list_var[i].data)
                else:
                    source_list_tensor = list_tensor


        self.list_tensor = numpy.empty(len(source_list_tensor),dtype=type(source_list_tensor[0]))
        for i in range(len(source_list_tensor)):
            self.list_tensor[i] = source_list_tensor[i].clone()
        self.store_lens = []
        self.store_shapes = []
        for i in range(len(self.list_tensor)):
            shape = list(self.list_tensor[i].shape)
            length = int(numpy.prod(shape))
            self.store_shapes.append(shape)
            self.store_lens.append(length)
        self.dim = sum(self.store_lens)
        #print(self.need_flatten)
        if self.need_flatten:
            self.flattened_tensor = torch.zeros(self.dim)
            self.load_param_to_flatten()
        else:
            assert len(self.list_tensor[0])==self.dim
            self.flattened_tensor = self.list_tensor[0]
            assert hex(id(self.flattened_tensor)) == hex(id(self.list_tensor[0]))
        self.syncro = self.assert_syncro()
    def load_flatten(self):
        if self.need_flatten:
            self.load_flattened_tensor_to_param()
        else:
            #print("flattened {}".format(self.flattened_tensor))
            #print("list tensor {}".format(self.list_tensor[0]))
            #print("syncro {}".format(self.assert_syncro()))
            assert hex(id(self.flattened_tensor)) == hex(id(self.list_tensor[0]))
        return()

    def point_clone(self):
        out = point(list_tensor=self.list_tensor,pointtype=self.pointtype,need_flatten=self.need_flatten)
        return(out)

    def assert_syncro(self):
        # check that list_tensor and flattened_tensor hold the same value

        temp = self.flattened_tensor.clone()
        #assert hex(id(self.flattened_tensor)) == hex(id(self.list_tensor[0]))
        cur = 0
        for i in range(self.num_var):
            temp[cur:(cur + self.store_lens[i])].copy_(self.list_tensor[i].view(-1))
            cur = cur + self.store_lens[i]
        diff = ((temp - self.flattened_tensor) * (temp - self.flattened_tensor)).sum()
        if diff > 1e-6:
            out = False
        else:
            out = True
        if not self.need_flatten:
            assert hex(id(self.flattened_tensor)) == hex(id(self.list_tensor[0]))

        return (out)

    def load_flattened_tensor_to_param(self):
        assert self.need_flatten==True
        cur = 0
        for i in range(self.num_var):
            self.list_tensor[i].copy_(self.flattened_tensor[cur:(cur + self.store_lens[i])].view(self.store_shapes[i]))
            cur = cur + self.store_lens[i]

    def load_param_to_flatten(self):
        assert self.need_flatten==True
        cur = 0
        for i in range(self.num_var):
            self.flattened_tensor[cur:(cur + self.store_lens[i])].copy_(self.list_tensor[i].view(-1))
            cur = cur + self.store_lens[i]

    def clone_cast_type(self,precision_type):
        assert precision_type in ("torch.FloatTensor","torch.DoubleTensor")
        new_list_tensor = [None]*len(self.list_tensor)
        for i in range(len(new_list_tensor)):
            new_list_tensor[i] = self.list_tensor[i].type(precision_type)

        out_point = point(list_tensor=new_list_tensor,pointtype=self.pointtype,need_flatten=self.need_flatten)
        return(out_point)


