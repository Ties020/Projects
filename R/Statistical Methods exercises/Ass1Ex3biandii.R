
#B) 1:
prob_smaller_than_3 <- pnorm(3, mean = 0, sd = 1);
prob_bigger_than_neg0.5 <- pnorm(-0.5, mean = 0, sd = 1, lower.tail = FALSE);
prob_between_neg1_and_2 <- pnorm(2, mean = 0, sd = 1) - pnorm(-1, mean = 0, sd = 1); 


#B) 2: 
prob_smaller_than_3_2 <- pnorm(3, mean = 3, sd = 4);
prob_bigger_than_neg0.5_2 <- pnorm(-0.5, mean = 3, sd = 4, lower.tail = FALSE);
prob_between_neg1_and_2_2 <- pnorm(2, mean = 3, sd = 4) - pnorm(-1, mean = 3, sd = 4); 

#finds the value for which 95% of the outcomes is smaller 
value_95_percentile <- qnorm(0.95, mean = 3, sd = 4)

#what table???