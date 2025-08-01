# Actor-critic for dynamic pricing


## Data
Simulated dynamic pricing (Uber as an example), including driver availability, loyalty tier, location, rewards. 


## Model
- Actor: a = pi(s)
- Critic: Q(s, a)
- advantage a - Q(s,a) represents how much we exceed critic’s expectioon
- update actor and critic


## Training
Following metrics are logged during training:
- Critic loss on test data
- Critic loss on training data
- Actor loss

Critic loss keeps decreasing and then stablize. Actor loss fluctuates around zero before converging. 

## Evaluation
Evaluation is tricky as we have two models.
Method 1: evaluate the critic using loss. Evaluate actor by comparing to baseline (comparing Q(s, a_validation) versus Q(s, actor(s))). 
Method 2: compare feature importance. 

## TODO 
debug feature importance. 


