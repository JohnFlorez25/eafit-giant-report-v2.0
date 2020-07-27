import Pkg; 
Pkg.add("Pkg")

Pkg.build("CodecZlib")
Pkg.add("VegaDatasets")
using VegaDatasets

Pkg.add("DataFrames")
using DataFrames

C = DataFrame(VegaDatasets.dataset("cars"))

dropmissing!(C)
M = Matrix(C[:,2:7])
names(C)

Pkg.add("MLBase")
using MLBase

car_origin = C[:,:Origin]
carmap = labelmap(car_origin) #from MLBase
uniqueids = labelencode(carmap,car_origin)

#UTILIZANDO TÉCNICAS PARA LA REDUCCIÓN DE DIMENSIONALIDADES

Pkg.add("StatsBase")
Pkg.add("Statistics")
using StatsBase
using Statistics

# center and normalize the data
data = M
data = (data .- mean(data,dims = 1))./ std(data,dims=1)

# cada auto es ahora una columna, PCA toma características por matriz de muestras
data'

Pkg.add("MultivariateStats")
using MultivariateStats

p = fit(PCA,data',maxoutdim=2)

P = projection(p)

P'*(data[1,:]-mean(p))

Yte = MultivariateStats.transform(p, data')

# reconstruir observaciones de prueba (aproximadamente)
Xr = reconstruct(p, Yte)

Pkg.add("LinearAlgebra")
using LinearAlgebra

norm(Xr-data')

Pkg.add("Plots")
using Plots

Plots.scatter(Yte[1,:],Yte[2,:],size=(450,300))

Plots.scatter(Yte[1,car_origin.=="USA"],Yte[2,car_origin.=="USA"],color=1,label="USA")
Plots.xlabel!("pca component1")
Plots.ylabel!("pca component2")
Plots.scatter!(Yte[1,car_origin.=="Japan"],Yte[2,car_origin.=="Japan"],color=2,label="Japan")
Plots.scatter!(Yte[1,car_origin.=="Europe"],Yte[2,car_origin.=="Europe"],color=3,label="Europe", size=(450,300))

p = fit(PCA,data',maxoutdim=3)
Yte = MultivariateStats.transform(p, data')
scatter3d(Yte[1,:],Yte[2,:],Yte[3,:],color=uniqueids,legend=false,size=(450,300))

#T-SNE
Pkg.add("ScikitLearn")
using ScikitLearn

@sk_import manifold : TSNE
tfn = TSNE(n_components=2) #,perplexity=20.0,early_exaggeration=50)
Y2 = tfn.fit_transform(data);
Plots.scatter(Y2[:,1],Y2[:,2],color=uniqueids,legend=false,size=(450,300),markersize=3)

#Adjuntamos y usamos las bibliotécas necesarias de UMAP
Pkg.add("UMAP")
using UMAP

L = cor(data,data,dims=2)
embedding = umap(L, 2)

Pkg.add("Distances")
using Distances

L = pairwise(Euclidean(), data, data,dims=1) 
embedding = umap(-L, 2)

Plots.scatter(embedding[1,:],embedding[2,:],color=uniqueids,size=(450,300))



