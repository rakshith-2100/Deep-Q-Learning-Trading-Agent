# Deep-Q-Learning-Trading-Agent
## Datset
Data of NIFTY50 stocks were collected from Jan 2010 to 2019 from this webiste (https://finance.yahoo.com/quote/%5ENSEI/history?period1=1262304000&period2=1559347200&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true) The dataset was cleaned thoroughly and the opening price was considered as the daily representive prices 
## State Representation 
State function here generates a state representation based on the given dataset, time(this reprents the current day) and window size(This is a hyperparmeter that represents the number of historic prices that will be considered here it is set to 10). The obejctive here is to consider the relative change in stock prices over a defined window size, emphasizing recent price movements. State function calculates the price difference between consecutive days and applying sigmoid to represent values in range of 0 to 1 (this is done because we would have some problems in the future dealing with 0, positive values in the later part the code).
## Reward function
Reward function is a simple representation of the profit that we gain due to the actions we take:
$$\ reward=max(sellprice-buyprice,0)$$
(Once again only positive values are only considered in a reward)
## Deep Q Learing in trading
We are using Deep Q learning algorithm to create the RL agent and then train it. The basic objective of a Q-learning/Deep Q-learning  is to maximize the returns $\ G_t$ i.e the net reward in a long run in Q_learning we calculate something called Q_value which is the expected Return at a given state for a given action 
$$\ Q(s,a) = E[G_t|S_t=s,A_t=a] $$ 
This Q_value determines what action should be taken based on the maximum value of Q 
In deep Q Learning, deep neural network is used to approximate the values of Q
