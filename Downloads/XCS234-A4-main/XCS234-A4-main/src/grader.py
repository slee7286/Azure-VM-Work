#!/usr/bin/env python3
import unittest
import random
import sys
import copy
import argparse
import inspect
import collections
import os
import pickle
import gzip
from graderUtil import graded, CourseTestRunner, GradedTestCase
import numpy as np
from itertools import product
import gymnasium as gym
from util import np2torch
from data import load_data
import torch
import torch.nn as nn
import torch.distributions as D

import submission

# Dimensions for Hopper hardcoded so we don't have to install MuJoCo on the autograder
OBSERVATION_DIM = 11
ACTION_DIM = 3
RANDOM_BATCH_SIZE = 100
SEQUENCE_LEN = 25

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################


def count_params(module_or_param_iter, include_non_grad=False):
    param_iter = (
        module_or_param_iter.parameters()
        if isinstance(module_or_param_iter, nn.Module)
        else module_or_param_iter
    )
    total = 0
    for p in param_iter:
        if p.requires_grad or include_non_grad:
            total += p.numel()
    return total


def fix_random_seeds(seed=123, set_system=True, set_torch=True):
    """
    Fix random seeds for reproducibility.
    Parameters
    ----------
    seed : int
        Random seed to be set.
    set_system : bool
        Whether to set `np.random.seed(seed)` and `random.seed(seed)`
    set_torch : bool
        Whether to set `torch.manual_seed(seed)`
    """
    # set system seed
    if set_system:
        random.seed(seed)
        np.random.seed(seed)

    # set torch seed
    if set_torch:
        torch.manual_seed(seed)


#########
# TESTS #
#########

class Test_2d(GradedTestCase):

    def setUp(self):
        # Hopper spaces
        self.observation_space = gym.spaces.Box(
            shape=[OBSERVATION_DIM], low=-np.inf, high=np.inf
        )
        self.action_space = gym.spaces.Box(
            shape=[ACTION_DIM], low=-1.0, high=1.0
        )

    # Returns a batch of random observations and actions
    def random_inputs(self, torchify=True, sequence=False):
        observations, actions = [], []
        for _ in range(RANDOM_BATCH_SIZE):
            if sequence:
                obs = [self.observation_space.sample() for _ in range(SEQUENCE_LEN)]
                act = [self.action_space.sample() for _ in range(SEQUENCE_LEN)]
            else:
                obs = self.observation_space.sample()
                act = self.action_space.sample()
            observations.append(obs)
            actions.append(act)
        observations = np.array(observations)
        actions = np.array(actions)
        if torchify:
            observations = np2torch(observations)
            actions = np2torch(actions)
        return observations, actions

    def setup_reward_model(self, hidden_dim=64):
        r_min, r_max = 0, 1

        fix_random_seeds(seed=int(234))
        self.reward_model = submission.RewardModel(OBSERVATION_DIM, ACTION_DIM, hidden_dim, r_min, r_max)

        fix_random_seeds(seed=int(234))
        self.ref_reward_model = self.run_with_solution_if_possible(submission, lambda sub_or_sol: sub_or_sol.RewardModel(OBSERVATION_DIM, ACTION_DIM, hidden_dim, r_min, r_max))

    @graded(timeout=1)
    def test_0(self):
        """2d-0-basic: test reward model optimizer"""
        self.setup_reward_model()
        self.assertTrue(hasattr(self.reward_model, 'optimizer'),
                        'Reward model has no optimizer attribute')
        self.assertIsInstance(self.reward_model.optimizer, torch.optim.AdamW)

    @graded(timeout=1)
    def test_1(self):
        """2d-1-basic: test the reward model for consistent network and optimizer parameters"""
        self.setup_reward_model()
        model_params = count_params(self.reward_model.net, include_non_grad=False)
        nparams_optimizer = count_params(self.reward_model.optimizer.param_groups[0]["params"])
        self.assertEqual(model_params, nparams_optimizer)

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded(timeout=1)
    def test_3(self):
        """2d-3-basic: test the computed reward is within [r_min, r_max] """
        self.setup_reward_model()
        obs, acts = self.random_inputs(torchify=False)
        for i in range(RANDOM_BATCH_SIZE):
            o, a = obs[i], acts[i]
            reward = self.reward_model.compute_reward(o, a)
            self.assertTrue(self.reward_model.r_min <= reward <= self.reward_model.r_max)

    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_3a(GradedTestCase):

    def setUp(self):
        # Hopper spaces
        self.observation_space = gym.spaces.Box(
            shape=[OBSERVATION_DIM], low=-np.inf, high=np.inf
        )
        self.action_space = gym.spaces.Box(
            shape=[ACTION_DIM], low=-1.0, high=1.0
        )

    # Returns a batch of random observations and actions
    def random_inputs(self, torchify=True, sequence=False):
        observations, actions = [], []
        for _ in range(RANDOM_BATCH_SIZE):
            if sequence:
                obs = [self.observation_space.sample() for _ in range(SEQUENCE_LEN)]
                act = [self.action_space.sample() for _ in range(SEQUENCE_LEN)]
            else:
                obs = self.observation_space.sample()
                act = self.action_space.sample()
            observations.append(obs)
            actions.append(act)
        observations = np.array(observations)
        actions = np.array(actions)
        if torchify:
            observations = np2torch(observations)
            actions = np2torch(actions)
        return observations, actions

    def setup_models(self, *models):
        models = set(models)
        kwargs = {
            'obs_dim': OBSERVATION_DIM,
            'action_dim': ACTION_DIM,
            'hidden_dim': 64,
            'segment_len': SEQUENCE_LEN,
            'lr': 1e-3
        }
        if 'base' in models:
            fix_random_seeds(seed=int(234))
            self.base_model = submission.ActionSequenceModel(**kwargs)
            fix_random_seeds(seed=int(234))
            self.ref_base_model = self.run_with_solution_if_possible(submission, lambda sub_or_sol: sub_or_sol.ActionSequenceModel(**kwargs))

    @graded(timeout=1)
    def test_0(self):
        """3a-0-basic: test action sequence optimizer"""
        self.setup_models('base')
        self.assertTrue(hasattr(self.base_model, 'optimizer'))
        self.assertTrue(isinstance(self.base_model.optimizer, torch.optim.AdamW))

    @graded(timeout=1)
    def test_1(self):
        """3a-1-basic: test the action sequence model for consistent network and optimizer parameters"""
        self.setup_models('base')
        model_params = count_params(self.base_model.net, include_non_grad=False)
        nparams_optimizer = count_params(self.base_model.optimizer.param_groups[0]["params"])
        self.assertEqual(model_params, nparams_optimizer)

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded(timeout=5)
    def test_3(self):
        """3a-3-basic: test action sequence forward output shapes """
        self.setup_models('base')
        obs, _ = self.random_inputs()
        mean, std = self.base_model.forward(obs)

        self.assertEqual(mean.shape, (len(obs), self.base_model.segment_len, self.base_model.action_dim), 'Means shape do not match')

        self.assertEqual(std.shape, (len(obs), self.base_model.segment_len, self.base_model.action_dim), 'Standard deviations shape do not match')

    @graded(timeout=5)
    def test_4(self):
        """3a-4-basic: test action sequence forward output is within the given boundaries """
        self.setup_models('base')
        obs, _ = self.random_inputs()
        mean, std = self.base_model.forward(obs)

        self.assertTrue(torch.logical_and(-1 <= mean, mean <= 1).all(),
                        'Mean values should be bounded in [-1, 1]')

        self.assertTrue(torch.logical_and(submission.LOGSTD_MIN <= torch.log(std), torch.log(std) <= submission.LOGSTD_MAX).all(),
                        f'Log STD values should be bounded in [{submission.LOGSTD_MIN}, {submission.LOGSTD_MAX}]')

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded(timeout=5)
    def test_6(self):
        """3a-6-basic: test action sequence model distribution type"""
        self.setup_models('base')
        obs, _ = self.random_inputs(sequence=True)
        obs = obs[:,0]
        distr = self.base_model.distribution(obs)

        self.assertTrue(type(distr) is D.Independent and type(distr.base_dist) is D.Normal,
                        'Incorrect distribution type')
    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded(timeout=5)
    def test_8(self):
        """3a-8-basic: test action sequence model act"""
        self.setup_models('base')
        obs, _ = self.random_inputs(torchify=False)
        obs = obs[0]
        action = self.base_model.act(obs)

        self.assertTrue(np.logical_and(-1 <= action, action <= 1).all(),
                        'Action values should be bounded in [-1, 1]')

    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_3b(GradedTestCase):

    def setUp(self):
        # Hopper spaces
        self.observation_space = gym.spaces.Box(
            shape=[OBSERVATION_DIM], low=-np.inf, high=np.inf
        )
        self.action_space = gym.spaces.Box(
            shape=[ACTION_DIM], low=-1.0, high=1.0
        )

    # Returns a batch of random observations and actions
    def random_inputs(self, torchify=True, sequence=False):
        observations, actions = [], []
        for _ in range(RANDOM_BATCH_SIZE):
            if sequence:
                obs = [self.observation_space.sample() for _ in range(SEQUENCE_LEN)]
                act = [self.action_space.sample() for _ in range(SEQUENCE_LEN)]
            else:
                obs = self.observation_space.sample()
                act = self.action_space.sample()
            observations.append(obs)
            actions.append(act)
        observations = np.array(observations)
        actions = np.array(actions)
        if torchify:
            observations = np2torch(observations)
            actions = np2torch(actions)
        return observations, actions

    def setup_models(self, *models):
        models = set(models)
        kwargs = {
            'obs_dim': OBSERVATION_DIM,
            'action_dim': ACTION_DIM,
            'hidden_dim': 64,
            'segment_len': SEQUENCE_LEN,
            'lr': 1e-3
        }

        if 'SFT' in models:
            fix_random_seeds(seed=int(234))
            self.sft_model = submission.SFT(**kwargs)
            fix_random_seeds(seed=int(234))
            self.ref_sft_model = self.run_with_solution_if_possible(
                submission, lambda sub_or_sol: sub_or_sol
            ).SFT(**kwargs)

    @graded(timeout=5)
    def test_0(self):
        """3b-0-basic: test sft update of model parameters"""
        self.setup_models('SFT')
        obs, actions = self.random_inputs(sequence=True)
        obs = obs[:,0]

        previous_params = list(self.sft_model.parameters())[0].clone()
        _ = self.sft_model.update(obs, actions)
        current_params = list(self.sft_model.parameters())[0].clone()

        self.assertFalse(torch.equal(previous_params.data, current_params.data))

    ### START_HIDE ###

    @graded(is_hidden=True, timeout=5)
    def test_1(self):
        """3b-1-hidden: test sft update"""
        self.setup_models('SFT')
        obs, actions = self.random_inputs(sequence=True)
        obs = obs[:,0]
        loss = self.sft_model.update(obs, actions)
        ref_loss = self.ref_sft_model.update(obs, actions)
        self.assertAlmostEqual(loss, ref_loss, places=6)

    ### END_HIDE ###

class Test_3c(GradedTestCase):

    def setUp(self):
        # Hopper spaces
        self.observation_space = gym.spaces.Box(
            shape=[OBSERVATION_DIM], low=-np.inf, high=np.inf
        )
        self.action_space = gym.spaces.Box(
            shape=[ACTION_DIM], low=-1.0, high=1.0
        )

    # Returns a batch of random observations and actions
    def random_inputs(self, torchify=True, sequence=False):
        observations, actions = [], []
        for _ in range(RANDOM_BATCH_SIZE):
            if sequence:
                obs = [self.observation_space.sample() for _ in range(SEQUENCE_LEN)]
                act = [self.action_space.sample() for _ in range(SEQUENCE_LEN)]
            else:
                obs = self.observation_space.sample()
                act = self.action_space.sample()
            observations.append(obs)
            actions.append(act)
        observations = np.array(observations)
        actions = np.array(actions)
        if torchify:
            observations = np2torch(observations)
            actions = np2torch(actions)
        return observations, actions

    def setup_models(self, *models):
        models = set(models)
        kwargs = {
            'obs_dim': OBSERVATION_DIM,
            'action_dim': ACTION_DIM,
            'hidden_dim': 64,
            'segment_len': SEQUENCE_LEN,
            'lr': 1e-3
        }
        if 'SFT' in models:
            fix_random_seeds(seed=int(234))
            self.sft_model = submission.SFT(**kwargs)
            fix_random_seeds(seed=int(234))
            self.ref_sft_model = self.run_with_solution_if_possible(
                submission, lambda sub_or_sol: sub_or_sol
            ).SFT(**kwargs)
            
        if 'DPO' in models:
            kwargs['beta'] = 0.1
            kwargs['lr'] = 1e-6
            fix_random_seeds(seed=int(234))
            self.dpo_model = submission.DPO(**kwargs)
            fix_random_seeds(seed=int(234))
            self.ref_dpo_model = self.run_with_solution_if_possible(
                submission, lambda sub_or_sol: sub_or_sol
            ).DPO(**kwargs)

    @graded(timeout=5)
    def test_0(self):
        """3c-0-basic: test dpo update of the model parameters"""
        self.setup_models('SFT', 'DPO')
        obs, actions1 = self.random_inputs(sequence=True)
        _, actions2 = self.random_inputs(sequence=True)
        obs = obs[:,0]

        previous_params = list(self.dpo_model.parameters())[0].clone()
        loss = self.dpo_model.update(obs, actions1, actions2, self.sft_model)
        current_params = list(self.dpo_model.parameters())[0].clone()

        self.assertFalse(torch.equal(previous_params.data, current_params.data))

    ### BEGIN_HIDE ###
    ### END_HIDE ###


def getTestCaseForTestID(test_id):
    question, part, _ = test_id.split("-")
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ("Test_" + question):
            return obj("test_" + part)


if __name__ == "__main__":
    # Parse for a specific test
    parser = argparse.ArgumentParser()
    parser.add_argument("test_case", nargs="?", default="all")
    test_id = parser.parse_args().test_case

    assignment = unittest.TestSuite()
    if test_id != "all":
        assignment.addTest(getTestCaseForTestID(test_id))
    else:
        assignment.addTests(
            unittest.defaultTestLoader.discover(".", pattern="grader.py")
        )
    CourseTestRunner().run(assignment)
