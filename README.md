# RL-for-Networking-Resillience
Version:
Python >= 3.7.0

torch >= 1.4.0

Microservice-based architectures enable different aspects of web applications to be created and updated independently, even after deployment. Associated technologies such as service mesh provide application-level fault resilience through attribute configurations that govern the behavior of request - response service -- and the interactions among them -- in the presence of failures. While this provides tremendous flexibility, the configured values of these attributes -- and the relationships among them -- can significantly affect the performance and fault resilience of the overall application. Furthermore, it is impossible to determine the best and worst combinations of attribute values with respect to fault resiliency via testing, due to the complexities of the underlying distributed system and the many possible attribute value combinations,  In this paper, we present a model-based reinforcement learning approach towards  service mesh fault resilience.  Our approach enables the prediction of the most significant fault resilience behaviors at a web application-level, scratching from single service to multi-service management with efficient agent collaborations.

NNk8s is trained network paramaters for k8s simulation model.

modelp.py is code for training k8s simulation model.

manual.py is complementary function for modelp.py, including dataloader set-up.

de_mDQN_thc.py is for training RL to enable online interaction between agent and environment. This version code is for thread-call double agent within dependent relationship.

function.py is complementary function for DQN, including reward definition.

extraction3.py is for cleaning data collected from actual istio experiments.
