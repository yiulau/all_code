def wrap_V_class_with_input_data(class_constructor,input_data):
    def wrapped_constructor(**kwargs):
        out = class_constructor(input_data=input_data,**kwargs)
        return(out)
    return(wrapped_constructor)