import numpy
import os
import pandas as pd

from distributions.logistic_regressions.logistic_regression import V_logistic_regression


class V_pima_inidan_logit(V_logistic_regression):
    def __init__(self,precision_type):
        dim = 10
        num_ob = 20
        y_np = numpy.random.binomial(n=1, p=0.5, size=num_ob)
        X_np = numpy.random.randn(num_ob, dim)
        #print(os.getcwd())
        #exit()
        #address1 = "/Users/patricklau/PycharmProjects/thesis_code/explain_hmc/input_data"
        #address2 = "/home/yiulau/work/thesis_code/explain_hmc/input_data"
        #address_list = [address1,address2]
        #for address in address_list:
         #   if os.path.exists(address):
         #       abs_address = address + "/pima_india.csv"

        abs_address = os.environ["PYTHONPATH"] + "/input_data/pima_india.csv"
        df = pd.read_csv(abs_address, header=0, sep=" ")
        # print(df)
        dfm = df.values
        # print(dfm)
        # print(dfm.shape)
        y_np = dfm[:, 8]
        y_np = y_np.astype(numpy.int64)
        X_np = dfm[:, 1:8]
        input_data = {"input":X_np,"target":y_np}
        super(V_pima_inidan_logit, self).__init__(input_data=input_data,precision_type=precision_type)

#vobj = V_pima_inidan_logit()