# LeWM Context

This context describes the language used by the LeWM repository for training and evaluating learned latent world models.

## Language

**LeWM**:
The project algorithm that learns latent visual dynamics and uses planning to choose actions.
_Avoid_: LeWorldModel when referring to the configured algorithm name

**JEPA**:
The neural dynamics model inside LeWM. It encodes observations into latent embeddings and scores candidate action sequences by predicted latent cost.
_Avoid_: Policy

**Policy**:
A training entrypoint represented by a LightningModule-like object under `source/policy`. A policy may expose an environment policy, but it is not necessarily the object passed directly to an environment.
_Avoid_: Model

**World Policy**:
The environment-facing policy object passed to `swm.World.set_policy`. For LeWM, this is a `stable_worldmodel` policy built from a planner and a JEPA model.
_Avoid_: Lightning policy

**Planner**:
The search procedure that proposes candidate action sequences and selects actions using the latent cost returned by JEPA.
_Avoid_: Predictor

**Candidate Action Sequence**:
A proposed rollout of future actions evaluated by the planner.
_Avoid_: Prediction

**Latent Cost**:
The score produced by comparing predicted latent embeddings against goal latent embeddings. Lower cost means the candidate action sequence is expected to move closer to the goal.
_Avoid_: Reward, action loss
