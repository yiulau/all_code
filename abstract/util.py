def wrap_V_class_with_input_data(class_constructor,input_data,prior_dict=None,model_dict=None):

    aux_dict = {}
    if not prior_dict is None:
        aux_dict.update({"prior_dict":prior_dict})
    if not model_dict is None:
        aux_dict.update({"model_dict":model_dict})

    def wrapped_constructor(**kwargs):
        aux_dict.update(kwargs)
        out = class_constructor(input_data=input_data,**aux_dict)
        return(out)

    return(wrapped_constructor)


# def contruct_fun(input_data,**kwargs):
#     print(kwargs)
#     return()
#
#
# input_data = {"x":1}
#
# model_data = {"y":13}
# prior_data = {"z":32}
# out = wrap_V_class_with_input_data(class_constructor=contruct_fun,input_data=input_data,model_dict=model_data,prior_dict=prior_data)
#
# out()



