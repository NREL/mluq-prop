import applications.KSE.main as main
import applications.KSE.util.data as data
import applications.KSE.util.simulation as simulation
import os

def make_base_input():
    inpt = {}
    inpt['Simulation name'] = "KS"
    inpt['Ndof'] = "128"
    inpt["Lx/pi"] = "32"
    inpt["epsilon_init"] = "0.0"
    inpt["ICType"] = "default"
    inpt["Tf"] = "550"
    inpt["Timestep"] = "0.25"
    inpt["Plot"] = "False"
    return inpt



def test_kse_no_src():
    inpt = make_base_input()
    Sim = data.simSetUp(inpt)
    Result = simulation.simRun(Sim)
    

def test_kse_src():
    inpt = make_base_input()
    inpt["SrcType"] = "dissRate"
    inpt["SrcCoeff"] = "1e-3"
    Sim = data.simSetUp(inpt)
    Result = simulation.simRun(Sim)


def test_kse_BNN_src():
    inpt = make_base_input()
    inpt["Tf"] = "10"
    inpt["SrcType"] = "mean_bnn"
    inpt["SrcCoeff"] = "5e-4"
    inpt["SrcNH"] = "2"
    inpt["SrcDX"] = "2"
    inpt["SrcDH"] = "5"
    inpt["SrcActivation"] =  "sigmoid"
    inpt["SrcPath"] =  os.path.join("../models/leanKSE/")
    inpt["SrcFCScale"] =  "5"
    inpt["SrcChiCFScale"] = "15"
    inpt["SrcNSamples"]  = "10"
    Sim = data.simSetUp(inpt)
    Result = simulation.simRun(Sim)

# if __name__ == "__main__":
#     test_kse_BNN_src()
