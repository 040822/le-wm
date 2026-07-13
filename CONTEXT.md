# LeWM Context

This context describes the language used by the LeWM repository for training and evaluating learned latent world models and world-action models.

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

**Value-JEPA LeWM**:
An experimental LeWM training policy that keeps the JEPA model and planner behavior unchanged while adding a ValueJEPA-style value loss during training.
_Avoid_: ValueJEPA paper reproduction

**Value Loss**:
An expectile temporal-difference loss over latent distances to a goal embedding, used as a training regularizer for Value-JEPA LeWM.
_Avoid_: Value head, reward model

**Window Goal**:
A goal embedding sampled from the same short training window that produced the current batch embeddings.
_Avoid_: Evaluation goal, future-goal dataset wrapper

**Fast-LeWAM**:
A world-action model that shares LeWM's visual latent space and can generate action chunks, predict their causal latent consequences, or do both jointly.
_Avoid_: LeWM, Fast LeWM

**Action Block**:
The group of consecutive environment actions represented by one action token. Its length is the dataset frameskip.
_Avoid_: Single environment action

**Action Horizon**:
The number of Action Blocks generated or evaluated together from one current latent.
_Avoid_: Number of environment steps

**Stage A Mode**:
The Fast-LeWAM behavior that generates an Action Horizon from the current latent.
_Avoid_: World-model mode

**Stage B Mode**:
The Fast-LeWAM behavior that predicts a latent sequence from the current latent and a clean Candidate Action Sequence.
_Avoid_: Action-generation mode

**Stage C Mode**:
The Fast-LeWAM ablation that jointly generates actions and predicts latent consequences in one causal world-action pass.
_Avoid_: Stage A+B

**Clean Action Estimate**:
The denoised action chunk inferred from a noisy action chunk and its predicted flow velocity.
_Avoid_: Ground-truth action

**Causal Prefix Latent Prediction**:
Parallel latent prediction in which each predicted future latent can depend only on the current latent and its corresponding action prefix.
_Avoid_: Autoregressive latent rollout
