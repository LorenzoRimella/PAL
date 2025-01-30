############
# This script includes code from https://github.com/theresasophia/pomp-astic/blob/master/Data/rotavirus_data.R.
# The purpose of this script is to explain the data preprocessing differences between the PAL analysis and Stocks et al's analysis of German rotavirus data
# It firstly demonstrates how Stocks et al preprocess their data by rescaling counts according to assumed reporting rates by region, and then sum the data for country level counts.
# We then show how we produce the unscaled `raw data` object we use in our analysis by summing across regions _without_ any scaling.
# As such there is no invertible function between our data and Stocks data and hence no valid comparisons between likelihoods may be made.
# The authors thank Yize Hao, Aaron Abkemeier, and Ed Ionides for first flagging this issue - a similar check is conducted by the script https://github.com/yhaoum/PAL-check/blob/master/rotavirus/data/rotavirus_data.R
###########


###############################################
# In this code we calculate the weekly number of new rotavirus cases in Germany
# between the years 2001-2008 stratified by age, namely age groups 0-4, 5-59 and 60+, 
# scaled up the the underreporting rates as inferred in Weidemann et al. 2013. 
# The data loaded is available at the following Github repository:
# https://github.com/weidemannf/thesis_code/tree/master/Mode_inference_paperI
# Author: Theresa Stocks
# Date: June 28, 2017
#################################################




# init
rm(list = ls())
#setwd("~/Dropbox/two-age-strata/Felix_data/thesis_code-master/Mode_inference_paperI")

## [[]] used to cite columns of a data.frame
EFSdata <- function(n){
  EFSdat<-numeric(0)
  for (i in 1:n){
    direct<-paste("source/RotadataEFS200",i,".txt")
    rotadata <- read.table(file=direct)
    for (j in 1:min(52,length(rotadata))){EFSdat<-cbind(EFSdat,rotadata[[j]])}
  }
  return(EFSdat)
}

# number of age groups 1:5 children, 6:8 adults, 9:10 elderly 
# Actually, year = 8
nrow(EFSdata(8))


# we consider data form 2001-2008
year <- 8
child <- seq(1,5,by=1)
adult <- seq(6,8,by=1)
elderly <- seq(9,10, by=1)

EFS <- matrix(nrow=3,ncol= ncol(EFSdata(8)))
EFS[1,] <- colSums(EFSdata(year)[child,])
EFS[2,]<- colSums(EFSdata(year)[adult,])
EFS[3,] <- colSums(EFSdata(year)[elderly,])
EFS

WFSdata <- function(n){
  WFSdat<-numeric(0)
  for (i in 1:n){
    direct<-paste("source/RotadataWFS200",i,".txt")
    rotadata <- read.table(file=direct)
    for (j in 1:min(52,length(rotadata))){WFSdat<-cbind(WFSdat,rotadata[[j]])}
  }
  return(WFSdat)
}

WFS <- matrix(nrow=3,ncol= ncol(EFSdata(8)))
WFS[1,] <- colSums(WFSdata(year)[child,])
WFS[2,] <- colSums(WFSdata(year)[adult,])
WFS[3,] <- colSums(WFSdata(year)[elderly,])
WFS

# change of reporting behaviour in beginning of 2005, so from beg 2001- end 2004 
# constant and from beg 2005- end 2008
years_till_change <- 4
time_unit <- 52
time_change <- years_till_change* time_unit
before <- seq(1,time_change,by=1)
after <- seq(time_change+1, ncol(EFSdata(8)), by=1)

# averaged posterior density of fitted underreportings rates from Weidemann et al. 2013 before and after 2005
EFS_rep_before <- 0.19 # Before 2005 underreporting rate
EFS_rep_after <- 0.241 # After 2005 underreporting rate
WFS_rep_before <- 0.043 # Before is lower than After
WFS_rep_after <- 0.063

# cases of children without underreporting
child_no_rep <- matrix(nrow=2,ncol= ncol(EFSdata(8)))
child_no_rep[1,] <- round(c(EFS[1,][before]/EFS_rep_before,EFS[1,][after]/EFS_rep_after))
child_no_rep[2,]<- round(c(WFS[1,][before]/WFS_rep_before,WFS[1,][after]/WFS_rep_after))
colSums(child_no_rep)

# cases of adult without underreporting
adult_no_rep <- matrix(nrow=2,ncol= ncol(EFSdata(8)))
adult_no_rep[1,] <- round(c(EFS[2,][before]/EFS_rep_before,EFS[2,][after]/EFS_rep_after))
adult_no_rep[2,] <- round(c(WFS[2,][before]/WFS_rep_before,WFS[2,][after]/WFS_rep_after))
colSums(adult_no_rep)

# cases of elderly without underreporting
elderly_no_rep <- matrix(nrow=2,ncol= ncol(EFSdata(8)))
elderly_no_rep[1,] <- round(c(EFS[3,][before]/EFS_rep_before,EFS[3,][after]/EFS_rep_after))
elderly_no_rep[2,] <- round(c(WFS[3,][before]/WFS_rep_before,WFS[3,][after]/WFS_rep_after))
colSums(elderly_no_rep)

## Below is the data used in the analysis of Stocks et al.
rotavirus <- data.frame(time = seq(1:ncol(EFSdata(8))),
                        cases1 = colSums(child_no_rep),
                        cases2 = colSums(adult_no_rep),
                        cases3 = colSums(elderly_no_rep))

########
# PAL data processing
# PAL data consists of total raw counts over all of Germany stratified by age  (EFS + WFS):
########

real_dat_PAL <- t(EFS + WFS)
# One can check this is consistent with the data `realdat` used in our analysis:
load("real_rotavirus_metadata.Rdata")
all(real_dat_PAL == realdat) # Returns True
