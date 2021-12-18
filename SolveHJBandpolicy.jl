# Solving the neoclassical growth model with a neural network
# Gustavo Mellior
# University of Liverpool
# G.Mellior@liverpool.ac.uk

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using ForwardDiff
import ModelingToolkit: Interval, infimum, supremum

@parameters x
@variables u(..), s(..)

Dx   = Differential(x)
# Parameters
σ     = 2
ρ     = 0.05
δ     = 0.05
α     = 0.34
Aprod = 1
kss   = (α*Aprod/(ρ+δ))^(1/(1-α))
VandC = 0 # Setting this to zero speeds up results
if VandC == 0
    # Solve for V then get c as a by-product of the neural network
    eq    = [(σ/(1-σ))*(max(Dx(u(x)),0.001))^((σ-1)/σ) + max(Dx(u(x)),0.001)*(Aprod*x^α-δ*x) - ρ*u(x) ~ 0]
else
    # Solve for V and c
    eq    = [(max(Dx(u(x)),0.001))^(-1/σ) - s(x) ~ 0.0,
            (σ/(1-σ))*((max(s(x),0.001))^(1-σ)) + max(Dx(u(x)),0.001)*(Aprod*x^α-δ*x) - ρ*u(x) ~ 0.0]
end
elems = size(eq)[1]

# Boundary condition
if VandC == 0
    bcs  = [Dx(u(kss)) ~ (Aprod*kss^α-δ*kss)^(-σ)]
else
    bcs  = [Dx(u(kss)) ~ (Aprod*kss^α-δ*kss)^(-σ); s(kss) ~ (Aprod*kss^α-δ*kss)]
end
# State space
domains           = [x ∈ Interval(kss*0.1,kss*1.5)]
# Neural network
neurons           = 20
chain             = [FastChain(FastDense(1,neurons,tanh)
                    ,FastDense(neurons,neurons,softplus)
                    ,FastDense(neurons,1)) for _ in 1:elems]
dx0               = kss*(1.5-0.1)/400

# Initial parameters of Neural network
initθ             = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))
discretization    = PhysicsInformedNN(chain, GridTraining(dx0),init_params =initθ)
if VandC == 1
    @named pde_system = PDESystem(eq,bcs,domains,[x],[u(x),s(x)])
else
    @named pde_system = PDESystem(eq,bcs,domains,[x],[u(x)])
end
prob              = discretize(pde_system,discretization)

const losses = []
cb = function (p,l)
    push!(losses, l)
    if length(losses)%500==0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

# Solve for the HJB equation and the FOC (if elems>1)
if elems>1
    maxit1 = 1200
    maxit2 = 4000
else
    maxit1 = 300
    maxit2 = 1000
end
res  = GalacticOptim.solve(prob, ADAM(0.06); atol=1e-7, cb = cb, maxiters=maxit1)
prob = remake(prob,u0=res.minimizer)
res  = GalacticOptim.solve(prob,BFGS(); cb = cb, maxiters=maxit2)
phi  = discretization.phi

## Load finite difference results and plot results
using DelimitedFiles
using LaTeXStrings
using Plots

# Load Finite difference results
FDM        = readdlm("NGMFDM.csv", ',', Float64)
vfunc      = FDM[:,1]
cons       = FDM[:,2]
HJBerr     = FDM[:,3]
plot_font  = "Computer Modern"
default(fontfamily=plot_font, linewidth=2, framestyle=:box)

kgrid      = LinRange(kss*0.1,kss*1.5,size(vfunc)[1])
I          = Int.(collect(size(res.minimizer)./elems))
# get V at points in kgrid
Vfunc(x)   = first(phi[1](x,res.minimizer[1:I[1]]))
if VandC == 1
    Cfunc(x)  = first(phi[2](x,res.minimizer[I[1]+1:end]))
end
# Plot V
plot(kgrid ,Vfunc.(kgrid),line =:dashdot,linewidth=3,label="Neural net",
    legend=:topleft,xlabel = L"k",ylabel=L"V(k)", fg_legend = :false, xtick=0:2:10, ytick=-20:1:-8)
display(plot!(kgrid,vfunc,label="Finite differences",legend=:topleft))

# Plot consumption
if VandC == 1
    plot(kgrid ,Cfunc.(kgrid),line =:dashdot,linewidth=3,label="Neural net",
        legend=:topleft,xlabel = L"k",ylabel=L"c(k)", fg_legend = :false, xtick=0:2:10, ytick=0.5:0.5:2)
else
    c          = (ForwardDiff.derivative.(Vfunc, Array(kgrid))).^(-1/σ)
    plot(kgrid,c,line =:dashdot,linewidth=3,label="Neural net",
        legend=:topleft,xlabel = L"k",ylabel=L"c(k)", fg_legend = :false)
end
Cfig       = plot!(kgrid,cons,label="Finite differences",legend=:topleft)
display(Cfig)
# png(Cfig,"Cfig.png")

# Compute HJB error
dVk          = ForwardDiff.derivative.(Vfunc, Array(kgrid))
HJBerror     = (σ/(1-σ))*((dVk).^((σ-1)/σ)) + dVk.*(Aprod*kgrid.^α-δ*kgrid) - ρ*Vfunc.(kgrid)
plot(kgrid,HJBerror,linewidth=3,ylims=(-1.2e-4,1e-4),
xlabel = L"k",ylabel="HJB error",label="Neural net", fg_legend = :false)
HJBerrorsfig = plot!(kgrid,HJBerr,label="Finite differences",legend=:topleft)
display(HJBerrorsfig)
# png(HJBerrorsfig,"HJBerrorsfig.png")

#Plot loss
display(plot(losses,legend = false, yaxis=:log))
