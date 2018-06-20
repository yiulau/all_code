from input_data.convert_data_to_dict import get_data_dict

permissible_vals = ["pima_indian","boston","subset_mnist","subset_cifar10","logistic_mnist","logistic_cifar10","logistic_8x8mnist"]
permissible_vals += ["australian","german","heart","diabetes","breast","8x8mnist"]


# out = get_data_dict("logistic_cifar10")
#
# print(out["input"].shape)
# exit()


for name in permissible_vals:
    out = get_data_dict(name)
    print("input shape {}".format(out["input"].shape))
    print("target shape {}".format(out["target"].shape))
    print("file {} success".format(name))
