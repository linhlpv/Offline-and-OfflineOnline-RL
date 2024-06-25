This is the flow of the MOPO implementation. It could be generalized to the code based flow for Model based Offline RL.

Dynamics contains the implemetation for dynamic model
Modules contains all of the implemetations for MLP, EnsembleLinear, Actor (both Stochastic and Deterministic), Critic and Transition model. 

The flow of transition model implementation: 
- EnsembleDynamics => EnsembleDynamicsModel

The flow of Actor:
- Distribution => ActorProb