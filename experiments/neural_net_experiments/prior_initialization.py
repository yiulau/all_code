# compare initializtion strategies
#
# experiment 1 prior with no hyperparameters (prior maybe model dependent , num of hidden units for example)

# option 1 initialize at optimized point with posterior

# option 2 initialize by generating from prior - not always possible (can do for horseshoe prior)

# option 3 initialize from uniform [-2,2]

# direct student t distribution with shape alpha
# experiment 2 hierarchical prior normal-inv_gamma, horseshoe
# option 4 initialize at 0 for network weight. initialize from prior for hyperparameter

# option 5 intialize at 0 for network weight, intialize from [-2,2] for hyperparameter
# short chains , dual averaging choose epsilon , gnuts

# plot energy

