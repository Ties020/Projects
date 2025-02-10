vlbw <- read.csv("vlbw.csv"); 
x <- vlbw$bwt;
par(mfrow = c(1, 3));
hist(x, xlab = "Weight in grams", ylab = "Number of babies"); 
qqnorm(x);
boxplot(x, ylab = "Weight in grams");

