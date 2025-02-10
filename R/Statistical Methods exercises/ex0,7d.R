func7d = function() {
  v = vector("numeric", length = 200)
  for(i in 1:200){
    sample_data <- sample(1:90, 40, replace = FALSE)
    v[i] <- median(sample_data)
  }
  v
}