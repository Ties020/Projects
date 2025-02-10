source("function02.txt");
absdiff = numeric(1000)
for(i in 1:1000){
  absdiff[i] = diffdice(1);
}
probability <- mean(absdiff == 3); #absdiff == 3 creates a logical vector of TRUE(1) and FALSE(0) where TRUE is
                                   #given if the value in absdiff is 3. Then the mean is calculated which is between 
                                   #0 and 1, in this case the probabilty of the difference being 3. 


source("function02.txt");
absdiff = numeric(1000)
for(i in 1:1000){
  absdiff[i] = diffdice(1);
}

expected_value <- mean(absdiff);
