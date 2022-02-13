# L3F
This is an parking occupancy prediction approach for poor-data parking lots considering data security. (unpublished, manuscript is not available now)

Abstract: Parking occupancy prediction (POP) can be used in many real-time parking-related services to significantly reduce the unnecessary cruising for parking and additional congestion. However the development of real-time POP in citywide Smart Parking Systems(SPS) has been stuck into a dilemma. On the one hand, SPS requires to address the challenge of implementing POP in the poor-data parking lots who need data supports; on the other hand, exchanging raw parking records has a risk of privacy leakage, which attracts increasingly attentions all around the world. To give an answer of the dilemma, this paper proposes a novel approach, named L3F, which contains two major ideas, namely 1) Learning to Learn: to select useful "guiders" and learn prior knowledge from them; and 2) Loose Federation: to form a federation to bridge data islands under the constraint of privacy protection. Specifically, we build two modules, Learner and Selector, under a Federated Learning (FL) framework. Accordingly, Learner is used to learn prior knowledge by a model pre-training process based on federated First-order Model-agnostic Meta-learning (FedFOMAML) mechanism, and Selector is used to select useful "guider" for Learner which is trained by Asynchronous Advantage Actor-Critic (A3C). Finally, the proposed approach is deployed on a real-world case study with 4 poor-data and 30 full-data parking lots in Guangzhou city, China. As shown by the results in target tasks, L3F can  1) reduce over 40% prediction errors, 2) fast adapt personalized training, 3) stable performance with small variance, 4) privacy protection by exchanging indirect data only. <br>

<div align="center"><img src="https://user-images.githubusercontent.com/49360609/147719583-a9787950-635d-4015-b1e2-2ed145260ab4.jpg" width="800"/>
</div>
<div align="center">Fig. 1. The overall structure of L3F
</div><br>
  

<div align="center">___________________________ case study __________________________
</div><br>

A common dataset with a minimum resolution of 5 minutes is created based on the parking occupancy data of 34 parking lots located in Guangzhou, China, from June 1 to 30, 2018. There are four parking occupancy prediction tasks with only 6 days parking records (June 19-24). The external condition that can be used is a federation with 30 members, who have more records but unwilling to share. The objective is to predict the parking occupancy in 30 minutes as accurately as possible in the last six days (25-30) of June 2018.

<div align="center"><img src="https://user-images.githubusercontent.com/49360609/148180675-41fc13b7-9a1d-46cf-9ffe-e633f2048804.jpg" width="600"/>
</div>
<div align="center">Fig. 2. Spatial information for parking lots: (a) Map of parking lots distribution; (b) Heat map of POI kernel density.
</div><br>

<div align="center"><img src="https://user-images.githubusercontent.com/49360609/148182865-4f4c872d-0cd6-4845-9f58-8f3b1298ce23.png" width="400"/>
</div><br>

**You can simply reproduce the testing results by running 'L3F.py'.**

