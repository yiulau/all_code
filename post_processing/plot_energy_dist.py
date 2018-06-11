import numpy
import matplotlib.pyplot as plt
def plot_energy_distribution(energy_vec):
    centered_energy = energy_vec - numpy.mean(energy_vec)
    diff_energy = energy_vec[1:]-energy_vec[:len(energy_vec)-1]
    #print(centered_energy)
    #exit()
    bins = numpy.linspace(-10, 10, 100)
    plt.hist(x=centered_energy, bins= bins,facecolor='g', alpha=0.5)
    #print("h")
    plt.hist(x=diff_energy,bins=bins, facecolor='r', alpha=0.5)
    plt.show()
    return()