# functions to process output from sampler object
import math
from adapt_util.objective_funs_util import ESJD
from post_processing.ESS_nuts import ess_stan
def get_min_ess_and_esjds(ran_sampler):

    ran_sampler.remove_failed_chains()
    sampler_diag_check = {"num_chains_removed":ran_sampler.metadata.num_chains_removed,
                          "num_restarts":ran_sampler.metadata.num_restarts}

    if ran_sampler.metadata.num_chains_removed ==0:
        samples_combined = ran_sampler.get_samples(permuted=True)
        esjd  = ESJD(samples_combined)
        esjd_normalized = esjd/math.sqrt(ran_sampler.metadata.average_num_transitons)
        ess = ess_stan(ran_sampler.get_samples(permuted=False))
        min_ess = min(ess)

        out = {"min_ess":min_ess,"esjd":esjd,"esjd_normalized":esjd_normalized}

    else:
        out = {"min_ess":0,"esjd":0,"esjd_normalized":0}

    out.update({"sampler_diag_check":sampler_diag_check})
    return(out)







