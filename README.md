# Implementing GAIL

Practice project for stepping into DL research projects

## Acquisitions from Implementation

-  in-depth read
- paper to code
- English writing

## Notes of GAIL

- **Intro** (contribution)

  - [GAIL (Generative Adversarial Imitation Learning)](https://arxiv.org/abs/1606.03476) proposed a <u>new general framework (part 3) for solving a sort of Imitation learning problem</u> that only offers a bundle of expert trajectories;
  - It also <u>instantiate an algorithm using a discriminator</u> (part 4, 5) that act like GAN.

- **Related Works**

  - Behavior Cloning
    - requires large amount of data
    - compounding error caused by covariate shift `(???)`
  - <u>Inverse Reinforcement Learning</u>
    - cost function `(???)` 
    - prioritize entire trajectories over others `(???)`
    - expensive to run (computational expensive) `(???)`
    - indirect yield actions (cost function as bridge layer)

- **Math Background** 

  - [IRL]
  - [key formula]

- **Method**

  - [method]
  - [insight]
    - [relation with TRPO]
    - [model free vs. model based]
    - [relation with GAN]
    - [harness generative adversarial training, fit distributions of states & actions defining expert behavior]

- **Experiment**

  - baseline: 

    - Behavior Cloning
    - Feature Expectation Matching (FEM)
    - Game-Theoretic Apprenticeship Learning (GTAL)

  - physic-based control tasks:

    [Cartpole, Acrobot, Mountain Car, HalfCheetah, Hopper, Walker, Ant, Humanoid, Reacher]

  - result

    - out performs basslines
    - reach good results

- **Interesting Parts & Drawbacks**

- **Future works**

## Implementation Structure

## Implementation Log (tech records)

## How to use

## Experiments