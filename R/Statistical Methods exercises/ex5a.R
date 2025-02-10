data <- read.table("Alice.txt") 

hours_alice <- data[,1]
hours_bob <- data[,2]

result <- t.test(hours_alice, hours_bob, alternative = "greater", conf.level = 0.99)

pval = result$p.value