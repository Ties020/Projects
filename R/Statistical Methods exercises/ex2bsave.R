observed_1 <- c(157, 69, 35, 19)
observed_2 <- c(130, 108, 35, 6)

chi_statistic_1 <- 47.86  #chi-squared test statistic for the first distribution, as calculated before


result_1 <- chisq.test(observed_1)  #perform chi-squared test for the first distribution
p_value_1 <- result_1$p.value

chi_statistic_2 <- 47.86  #chi-squared test statistic for the second distribution, same as first
result_2 <- chisq.test(observed_2)

p_value_2 <- result_2$p.value
