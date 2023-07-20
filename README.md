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
$$\ Q(s,a)=R_t(s,a)+γmax_{a'}(Q'(s',a')) \$$ <br>
The Bellman equation <br>
In the equation $\ s$ is the current state and $\ a\$ is the action taken and $\ s'$ is the next state and $\ R(s,a)$ is the reward that we get after doing action a and $\ γ$ here is the discount factor which is a hyperparameter(*in the code γ=0.95*)that tells how valuable is a reward which we get in a future state and $\ a'$ is the set of actions possible in the next state $\max_a'(Q'(s',a'))$ and  determines the maximum Q_value possible for the next state which is also calculated from the neural network.<br>
The Q_value that we get from the bellmen equation would serve as the target Q_value, so we call it as **Q_target** i.e $\ Q'(s,a)$  so the loss computed would be the mean square difference between q_value and q_target 
$$\ L(θ)=1/N\sum_{i∈N}(Q_θ(S_i,A_i)-Q'_θ(S_i,A_i))^2\$$ <br>
This loss would be used to fine tune the model i.e update weights $\ θ$ using Stocastic Gradient Descent optimizer(SGD)
$$\ θ <- θ-α\frac{∂L}{∂θ}$$
### Training a Deep Q model
Before training we look into the training of the model we have to know two concepts that play a major role in trainng, **epsilon greddy strategy** and **replay memory** .
#### Epsilon Greedy Strategy
This is used to maintain a balance in **exploration** and **exploitation**. Exploration basically means that the agent is trying to explore all the market conditions and try to learn more about the market and exploitation means that the agent is trying to exploit the market conditions thus maximizing the rewards. There should be a balance between exploration and exploitation such that our agent will be prepared for any situations and parallely maximize profits to any situation, hence we use epsilon greedy strategy. Initially the agent  should explore to know  more about the enivironment and learn important details about the market and slowly it should start exploiting to maximize returns to do this, we define **ε**( Epsilon ), the probablity that the agent would explore the environment rather than exploit it. This is initially set to 1 and this epsilon decays by some rate(which is a hyperparameter) as the agent becomes greddy with time. We choose a random number r ranging between 0 and 1 such that if $\ (r>ε)$ it would choose exploitation i.e choose action based on the maximum Q_value that we get from the model and if $\ (r<ε)$ then random actions are taken sure that agent can explore the environment<br>
*( This can be seen in a function called trade from the class called ModelandTrade in the code )*
#### Replay Memory
Before we know about replay memory we should know about **experience**( $\ e_t$). The agents experience is defined as a tuple
$$\ e_t=(s_t,a_t,r_{t+1},s_{t+1})$$
$\ s_t$ is the state at time t $\ a_t$ is the action taken on state $\ s_t$ and $\ r_{t+1}$ is the reward after taking action $\ a_t$ on state $\ s_t$ and $\ s_{t+1}$ is the next state.<br>
We get experiences at each time step from random exploration and exploitation using epsilon greedy strategy and all the agents experience in each timestep is stored in the replay memory.*( in the code it is called as memory )*.
#### Training 
The replay memory is used for training as random experiences will be taken and will be used to calculate the q_target,q_value hence computing the loss and optimizing it.
The dataset was split into batches ( *here the batch size was 32* ) and after each batch the replay memory records the experience and uses it to train the model and optimize it and then the Epsilon value ε is mutiplied with a decay rate such that epsilon gets reduced. The number of episodes were set to 1000 and the epsilon decay rate was set to 0.01 other<br>
*Note: In the replay memory in our code we also considered a boolean charecter that determines if the episode is terminated or not*
