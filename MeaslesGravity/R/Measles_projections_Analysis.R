library(Rcpp)
# load meta data 
load(file = 'data/datamat.Rdata')
load("data/UKMeaslesmetadata.Rdata")
sourceCpp('cpp/biweeklyaggregate.cpp')
sourceCpp("cpp/measles_sim.cpp")
births <- as.matrix(1*births, nrow= 468)

# set maximum PAL parameters
U <- 40
c = 0.92
a = 0.37
beta = 24.85
rho = 0.22
gamma = 0.95
g = 630.5
S_0 = 0.029
E_0 = 0.00099
I_0 = 0.00099
init_dist_props1 = rbind(rep(S_0,U), rep(E_0,U), rep(I_0,U), rep(1-S_0-E_0-I_0,U))
# get implied init distribution for projections
init <- PoislowGravity(p, datamat[-469,], t(t(init_dist_props1)*p),v_by_g, births, params = c(beta,rho,gamma) , q, survivalprob  , c, g , h = 1, a,m = 4, k = 40, ncores = 4, tstar = 100)

init_dist <- matrix(data = 0, nrow = 4, ncol = 40)
for (i in 1:40) {
  init_dist[,i]<- init$lambda[,i]/sum(init$lambda[,i])
}

t = 100
params<- c(beta,rho,gamma)

#setwd("~/Documents/PhD/multinomialapproxproj/R/Measles_plots")
library(matrixStats)
library(maps)
library(mapdata)
library(maptools)
library(rgdal)
library(ggmap)
library(ggplot2)
library(ggpubr)
library(rgeos)
library(broom)
library(plyr)
library(dplyr)
library(raster)
library(scales)
library(latex2exp)
if (!require(gpclib)) install.packages("gpclib", type="source")
gpclibPermit()
worldmap <- map_data('world')
load(file = 'measlesplottingdata.Rdata')
load(file = 'datamat.Rdata')



sim <- Sim_Gravity(p,init_dist_props1, v_by_g , births, params, q , survivalprob, t, n_steps = 4 , c , g, h = 1, a, m = 4, K = U)

simulations_20 <- matrix(data = 0, nrow = 1000, ncol = 40)
simulations_30 <- matrix(data = 0, nrow = 1000, ncol = 40)
simulations_40 <- matrix(data = 0, nrow = 1000, ncol = 40)
simulations_50 <- matrix(data = 0, nrow = 1000, ncol = 40)
for (i in 1:1000) {
  sim <- Sim_Gravity(p,init_dist, v_by_g , births, params, q , survivalprob, t, n_steps = 4 , c , g, h = 1, a, m = 4, K = U)
  simulations_20[i,] <- sim[,1]
  simulations_30[i,] <- sim[,2]
  simulations_40[i,] <- sim[,3]
  simulations_50[i,] <- sim[,4]
}



# create matrix with low quantiles, mean, and upper quantiles for the 4 time points we are predicting
predictive_data <- matrix(data = 0, nrow = 12, ncol = 40)

predictive_data[1,] = colQuantiles(simulations_20, probs = 0.05)
predictive_data[2,] = colQuantiles(simulations_30, probs = 0.05)
predictive_data[3,] = colQuantiles(simulations_40, probs = 0.05)
predictive_data[4,] = colQuantiles(simulations_50, probs = 0.05)
predictive_data[5,] = colMeans(simulations_20)
predictive_data[6,] = colMeans(simulations_30)
predictive_data[7,] = colMeans(simulations_40)
predictive_data[8,] = colMeans(simulations_50)
predictive_data[9,] = colQuantiles(simulations_20, probs = 0.95)
predictive_data[10,] = colQuantiles(simulations_30, probs = 0.95)
predictive_data[11,] = colQuantiles(simulations_40, probs = 0.95)
predictive_data[12,] = colQuantiles(simulations_50, probs = 0.95)

par(mfrow = c(2,2))
cols <- c('X1')
for (i in 2:12) {
  cols <- c(cols, paste("X", i, sep = ''))
}


for (i in 1:40) {
  predictive_data[,i] <- ((predictive_data[,i])/(p[i]*q[i])+1)
}



citydf <- data.frame(cbind(long_lat,cities, p, t(log((predictive_data)))))
#citydf <- mutate_at(citydf, cols, funs(log((.+ 1)/(q*p))))
lims = c(min(log((predictive_data))), max(log((predictive_data))))
b <- c(min(log((predictive_data))), median(log((predictive_data))), max(log((predictive_data))))
citydf <- citydf %>% rename(longitude = lon, latitude = lat)



## Try concentric circles
for (i in c(1,2,3,4)) {
  
  gg <- ggplot() + geom_polygon(data = worldmap, aes(x = long, y = lat, group = group), fill = "white", color = 'black', alpha = 0.75)
  gg <- gg + coord_fixed(1.3, xlim = c(-5.5,1.5), ylim = c(50, 55)) #This gives the map a nice 1:1 aspect ratio to prevent the map from appearing squashed
  gg <- gg + geom_point( data = citydf, aes(x = longitude, y = latitude, size = (p), color = !!as.name(cols[i])), alpha = 1)+ 
    geom_point( data = citydf, aes(x = longitude, y = latitude, size = (p)/((1/0.625)^2), color = !!as.name(cols[i+4])), alpha = 1)  +    
    geom_point( data = citydf, aes(x = longitude, y = latitude, size = (p)/16, color = !!as.name(cols[i+8])), alpha = 1) + scale_size_area(max_size = 20,name = 'population', guide = 'none') + 
    scale_colour_gradientn(
      labels = comma,
      limits = lims,
      breaks = b,
      colours = c('lightgreen', 'yellow','red')) +
    labs(x = 'longitude', y = 'latitude', size = 'log population', color = TeX(r"($\log \left(\frac{D_{k,t}}{n_k}+1\right)$ \phantom{test})")) +
    theme(legend.position = "bottom", legend.key.width = unit(1.5, 'cm'))
  
  
  
  assign(paste('gg',cols[i],sep=""),gg)
}


ggarrange(ggX1 + rremove('xlab'),ggX2+rremove('xlab') +rremove('ylab') ,ggX3,ggX4 + rremove('ylab'),
          common.legend = TRUE,
          legend = "top",
          align = 'hv',
          ncol = 2, nrow = 2)
