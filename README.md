# Deep-Q-Learning-Trading-Agent
## Datset
Data of NIFTY50 stocks were collected from Jan 2010 to 2019 from this webiste (https://finance.yahoo.com/quote/%5ENSEI/history?period1=1262304000&period2=1559347200&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true) The dataset was cleaned thoroughly and the opening price was considered as the daily representive prices 
## State Representation 
State function here generates a state representation based on the given dataset, time(this reprents the current day) and window size(This is a hyperparmeter that represents the number of historic prices that will be considered here it is set to 10). The obejctive here is to consider the relative change in stock prices over a defined window size, emphasizing recent price movements. State function calculates the price difference between consecutive days and applying sigmoid to represent values in range of 0 to 1 (this is done because we would have some problems in the future dealing with negetive values, 0 in the later part the code).
## Reward function
Reward function is a simple representation of the profit that we gain due to the actions we take:
$$\ reward=max(sellprice-buyprice,0)$$
(Once again only positive values are only considered in a reward)
## Deep Q Learing in trading
We are using Deep Q learning algorithm to create the RL agent and then train it. The basic objective of a Q-learning/Deep Q-learning  is to maximize the returns $\ G_t$ i.e the net reward inn a long run. In Q_learning we calculate Q_value which is the expected Return at a given state for a given action,
$$\ Q(s,a) = E[G_t|S_t=s,A_t=a] $$ <br>
This Q_value determines what action should be taken based on the maximum value of Q.<br>
In deep Q Learning, deep neural network is used to approximate the values of Q
![photo_2023-07-20_13-17-13](https://github.com/rakshith-2100/Deep-Q-Learning-Trading-Agent/assets/99346822/d609477e-a4a5-4192-8dc3-fc07cac04307) <br>
This image explains the working of the neural network and also tells us about the q_value that we get as an output.
The Q_value from the neural network is then compared with the Q value from the **Bellman equation**<br>
$$\ Q(s,a)=R_t(s,a)+γmax_a'(Q'(s',a')) \$$ <br>
The Bellman equation <br>
In the equation $\ s$ is the current state and $\ a\$ is the action taken and $\ s'$ is the next state and $\ R(s,a)$ is the reward that we get after doing action a and $\ γ$ here is the discount factor which is a hyperparameter that tells how valuable is a reward which we get in a future state and $\ a'$ is the set of actions possible in the next state $\max_a'(Q'(s',a'))$ and  determines the maximum Q_value possible for the next state which is also calculated from the neural network.<br>
The Q_value that we get from the bellmen equation would serve as the target Q_value, so we call it as **Q_target** i.e $\ Q'(s,a)$  so the loss computed would me the mean square difference between q_value and q_target 
$$\ L(θ)=1/N\sum_{i∈N}(Q_θ(S_i,A_i)-Q'_θ(S_i,A_i))^2\$$ <br>
This loss would be used to fine tune the model i.e update weights $\ θ$ using Stocastic Gradient Descent optimizer(SGD)
$$\ θ <- θ-α\frac{∂L}{∂θ}$$
### Training a Deep Q model
