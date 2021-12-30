# Load all packages
using Pkg; Pkg.activate("."); Pkg.instantiate()
using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, Optim
using DiffEqFlux, Flux
using Plots
gr()
using JLD2, FileIO
using Statistics

# Read data from pre-sorted CSV file
using CSV, DataFrames
df = CSV.File("/Users/shentianxiao/Desktop/Machine Learning/Final Project/Data_test.csv") |> DataFrame

#Sort All Data
t = df[!,"Column1"]
time = df[!,["Year","Month","Day","Hour"]]
weather = df[!,["tempC","windspeedKmph","precipMM","humidity", "visibility", "pressure", "DewPointC","WindGustKmph"]]
solar = df[!,["GHI", "cloudcover"]]
chemical = df[!,["O3_obj","RH_obj"]]
transport = df[!,["PM_t1","PM_t2","NOx_t1","NOx_t2","SO2_t1","SO2_t2"]]
AQI = df[!,["PM2.5_obj","NOx_obj","SO2_obj"]]
consumption = df[!,["Coal","Petroleum","Natural gas","Electricity"]]
traffic = df[!,"Accident Count"]
wind = df[!,["windspeedKmph","winddirDegree"]]
hour = df[!,["Hour"]]

AQI0 = [AQI[1,1],AQI[1,2],AQI[1,3]]
tspan = (1,length(t))
tsteps = range(tspan[1], tspan[2], length = length(t))

## Define the network
# Gaussian RBF as activation
rbf(x) = exp.(-(x.^2))

# Multilayer FeedForward
UE = FastChain(
    FastDense(6,5,rbf), FastDense(5,5, rbf), FastDense(5,5, rbf), FastDense(5,3)
)
UT = FastChain(
    FastDense(11,5,rbf), FastDense(5,5, rbf), FastDense(5,5, rbf), FastDense(5,3)
)
UR = FastChain(
    FastDense(7,5,rbf), FastDense(5,5, rbf), FastDense(5,5, rbf), FastDense(5,3)
)
UD = FastChain(
    FastDense(9,5,rbf), FastDense(5,5, rbf), FastDense(5,5, rbf), FastDense(5,1)
)
UD2 = FastChain(
    FastDense(9,5,rbf), FastDense(5,5, rbf), FastDense(5,5, rbf), FastDense(5,1)
)

# Get the initial parameters
pE = initial_params(UE)
pT = initial_params(UT)
pR = initial_params(UR)
pD = initial_params(UD)
pD2 = initial_params(UD2)
p=[pE;pT;pR;pD;pD2]

# Define the hybrid model
function ude_dynamics!(AQI, p, t, time,solar,weather,chemical,transport,consumption,traffic,wind,hour)
    pEt = p[1:length(pE)]
    pTt = p[(length(pE)+1),(length(pE)+length(pT))]
    pRt = p[(length(pE)+length(pT)+1):(length(pE)+length(pT)+length(pR))]
    pDt = p[(length(pE)+length(pT)+length(pR)+1):(length(pE)+length(pT)+length(pR)+length(pD))]
    pDt2 = p[(length(pE)+length(pT)+length(pR)+length(pD)+1):end]
    Qe = UE([[traffic consumption] hour], pEt) # Network prediction
    Qt = UT([[transport wind] AQI],pTt)
    Qr = UR([[solar chemical] AQI],pRt)
    Ed = UD([hour weather],pDt)
    Ed2 = UD2([hour weather],pDt2)
    dAQI = Qe + Qt + Qr
end

#nn_dynamics!(dAQI,AQI,time,solar,weather,chemical,transport,consumption,traffic,wind,hour,p,t) = ude_dynamics!(AQI,time,solar,weather,chemical,transport,consumption,traffic,wind,hour,p,t)
prob_nn = ODEProblem(ude_dynamics!, AQI0, tspan, p)

function predict(θ)
    Array(solve(prob_nn, Vern7(), p=θ, saveat = t,
                         abstol=1, reltol=1,
                         sensealg = ForwardDiffSensitivity()))
end

#Define the loss function
function loss(θ)
    pred = predict(θ)
    sum(abs2, AQI .- pred), pred
end

const losses = []
callback(θ,l,pred) = begin
    push!(losses, l)
    if length(losses)%50==0
        println(losses[end])
    end
    false
end

res1_uode = DiffEqFlux.sciml_train(loss, p, ADAM(0.01), cb=callback, maxiters = 100)
println("Training loss after $(length(losses)) iterations: $(losses[end])")
res2_uode = DiffEqFlux.sciml_train(loss, res1_uode.u, BFGS(initial_stepnorm=0.01), cb=callback, maxiters = 10000)
