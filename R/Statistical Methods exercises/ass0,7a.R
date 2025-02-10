add2=function(x){ 
  o = numeric(100)
  for(i in 1:length(x)){
    x[i] <- x[i] - 3
    o[i] <- x[i]^2
    }
    return(o)
}
c