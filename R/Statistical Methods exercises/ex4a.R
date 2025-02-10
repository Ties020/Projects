source("function02.txt");
absdiff = numeric(1000)
for(i in 1:1000){
  absdiff[i] = mean(diffdice(i));
}
second = 1.9444;

plot(absdiff, type = "o", cex = 0, xlab = "Number of trials");
abline(1.9444,0, col = "red");

