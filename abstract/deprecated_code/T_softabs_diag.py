import torch, numpy
from abstract.abstract_class_T import T
from abstract.abstract_class_point import point
class T_softabs_diag_e(T):
    def __init__(self,metric,linkedV):
        self.metric = metric
        super(T_softabs_diag_e, self).__init__(linkedV)

    def evaluate_scalar(self,q_point=None,p_point=None):
        if not q_point is None:
            self.linkedV.load_point(q_point)
        if not p_point is None:
            self.load_point(p_point)
        _, H_ = self.linkedV.getH_tensor()
        _, mdiagH = self.linkedV.getdiagH_tensor()
        mlambda,mLogdetmetric = self.fcomputeMetric(mdiagH)
        tau = 0.5 * torch.dot(mlambda * self.flattened_tensor, self.flattened_tensor)
        out = tau + 0.5 * mLogdetmetric
        return (out)


    def dtaudp(self,p_flattened_tensor, mlambda):
        out = mlambda * p_flattened_tensor
        return (out)

    def dtaudq(self,p_flattened_tensor, mdiagH, mlambda, mgraddiagH):
        # mdiagH = diagonal of hessian (2nd derivatives H_ii - vector
        # mgraddiagH = derivatives of diagonal of hessian - matrix
        msoftabsalpha = self.metric.msoftabsalpha
        mgradhelper = torch.zeros(len(p_flattened_tensor))
        for i in range(len(mgradhelper)):
            hlambda = mdiagH[i] * mlambda[i]
            v = 1.0 / mdiagH[i]
            if (abs(msoftabsalpha * mdiagH[i]) < 18):
                v += msoftabsalpha * (hlambda - 1.0 / hlambda)
            mgradhelper[i] = - 0.5 * v * p_flattened_tensor[i] * p_flattened_tensor[i]
        out = torch.mv(mgraddiagH, mgradhelper)


        return (out)

    def generate_momentum(self,q):
        _,mdiagH = self.linkedV.getdiagH_tensor(q)
        mlambda,_ = self.fcomputeMetric(mdiagH)
        out = point(list_tensor=self.list_tensor,pointtype="p",need_flatten=self.need_flatten)
        out.flattened_tensor.copy_(torch.randn(len(mlambda)) / torch.sqrt(mlambda))
        out.load_flatten()
        return (out)

    def dphidq(self,dV,mdiagH,mgraddiagH, mlambda):
        msoftabsalpha = self.metric.msoftabsalpha
        mgradhelper = torch.zeros(len(mdiagH))
        for i in range(len(mgradhelper)):
            hlambda = mdiagH[i] * mlambda[i]
            v = 1.0 / mdiagH[i]
            if (abs(msoftabsalpha * mdiagH[i]) < 18):
                v += msoftabsalpha * (hlambda - 1.0 / hlambda)
            mgradhelper[i] = 0.5 * v

        out = torch.mv(mgraddiagH,mgradhelper) + dV
        return(out)

    def fcomputeMetric(self,mdiagH):
        mlambda = torch.zeros(self.dim)
        msoftabsalpha = self.metric.msoftabsalpha

        for i in range(self.dim):
            lam = mdiagH[i]
            alphalambda = msoftabsalpha * lam


            if (abs(alphalambda) < 1e-4):
                mlambda[i] = msoftabsalpha * (1. - (1. / 3.) * alphalambda * alphalambda)
            elif (abs(alphalambda) > 18):
                mlambda[i] = 1 / abs(lam)
            else:
                mlambda[i] = numpy.tanh(msoftabsalpha * lam) / lam
        mlogdetmetric = 0

        mlogdetmetric = -torch.log(mlambda).sum()

        return (mlambda, mlogdetmetric)