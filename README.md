# Bayes-net-smart-building

# Model Used
Inhomogenous Markov chain

## Results
The initial model submitted on the 15th of November scored 79261 on the leaderboard with
a total runtime of 40 seconds. This initial model contained an early version of the transition
matrices and no reliability metrics were taken into account, with initial testing on
example_test.py scoring around 74000. A second updated model was submitted on the 19th
of November. This model included some reliability metrics and modifications to the transition
matrices to try to lower cost, with the example_test.py testing scoring around 67000. This
model scored 78786 on the second leaderboard with a runtime of 40 seconds, a very minor
improvement to our first model with the same speed. From this we can conclude that our
model is fast to run but it is insufficiently accurate.
