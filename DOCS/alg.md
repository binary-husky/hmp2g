# DQN


## estimate target $Q^t_{t}(a)$ of next t
$Q_{t+1}(a1_{t+1}, a2_{t+1},..) = NN(o_{t+1})$
$Q^{max}_{t+1} = max_a(Q_{t+1}(a1_{t+1}, a2_{t+1},..))$
$Q^{target}_{t} = r + gamma*Q^{max}_{t+1}(s_{t+1} | a_t)$

## correct and fix current estimation

$Q^{nn}_{t}(a1_{t+1}, a2_{t+1},..) = NN(o_{t})$
$Q^{nn}_{t}(a_t) = NN(o_{t})$
$loss = mse(Q^{target}_{t}, Q^{nn}_{t}(a_t))$
