import numpy as np
import utils
import json
import random
import ipdb
import sys
import math
import os
from itertools import chain

np.set_printoptions(precision=3)


def get_baseline(df, agent_id, num_agent, num_rounds, num_update_cycle, type=0):
    if type == 0:  # random
        return random.randint(2, 10) * 10

    if type == 1:  # decay
        total_cycle=math.floor(num_rounds/num_update_cycle)
        cycle_baseline=[]
        # Deline curve for baselines (i. linear)
        for i in range(total_cycle):
            if i == 0:
                baseline_ls=[]
                for j in range(num_agent):
                    ini_baseline = random.randint(20, 100)
                    baseline_ls.append(ini_baseline)
            de_baseline = np.array(baseline_ls)*(1-i/total_cycle)
            cycle_baseline.append(de_baseline)
        
        agents_baseline=[]

        # baselines during the cycles
        for j in range(total_cycle):
            round_baseline = []
            for i in range(num_agent):
                personal_baseline=np.repeat(cycle_baseline[j][i],num_update_cycle).tolist()
                round_baseline.append(personal_baseline)
            agents_baseline.append(round_baseline)
            
        # baselines during residual rounds
        personal_baseline_final=[np.repeat(0,(num_rounds % num_update_cycle)).tolist() for i in range(num_agent)]
        agents_baseline.append(personal_baseline_final)

        return np.array(list(chain(*np.array(agents_baseline)[:,agent_id])))


def generate_emission(df, agent_id, num_rounds, carbon_emission_baseline, type=0):
    if type == 0:  # random
        return np.random.random((1, num_rounds))

    if type == 1:  # brown motion
        mu = df["emission_brown"]["mu"]
        sigma = df["emission_brown"]["sigma"]
        emission = np.zeros(num_rounds)
        emission[0] = carbon_emission_baseline[agent_id][0] + random.gauss(0, 1)
        for i in range(1, num_rounds):
            emission[i] = emission[i - 1] * mu + sigma * random.gauss(0, 1)
        return emission

    if type == 2:  # sample
        sample_data = [
            [500, 400, 300, 800, 455, 651, 656],
            [400, 500, 200, 800, 452, 641, 654],
            [852, 422, 123, 122, 448, 123, 1256],
        ]
        return np.array(sample_data)[:, agent_id]


def get_coefficient(df, agent_id, round_id, data, type=0):
    if type == 0:  # random
        a = random.randint(0, 50)
        b = 50 - a
        c = random.randint(0, 50)
        d = 50 - c
        return a, b, c, d

    if type == 1:  # greedy
        if data[0] > data[1]:
            a = 50
        else:
            a = 0
        if data[2] > data[3]:
            c = 50
        else:
            c = 0
        b = 50 - a
        d = 50 - c
        return a, b, c, d

    if type == 2:  # sample
        sample_data = [
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [10, 20, 4, 5, 4, 45, 47],
                [40, 30, 46, 45, 46, 5, 3],
                [30, 0, 6, 45, 50, 4, 1],
                [20, 50, 44, 5, 0, 46, 49],
            ],
            [
                [45, 45, 40, 44, 38, 50, 1],
                [5, 5, 10, 6, 12, 0, 49],
                [45, 0, 0, 4, 1, 0, 4],
                [5, 50, 50, 46, 49, 50, 46],
            ],
        ]
        return np.array(sample_data)[round_id, :, agent_id]


def get_token_settlement(
    df: dict,
    agent_id: int,
    round_id: int,
    credit_list: list = [],
    score_list: list = [],
    quota: float = 0.0,
    type: int = 0,
):
    if type == 0:  # Random
        return random.randint(-10, 10)

    if type == 2:
        # init token + (Personal carbon credit*β)
        if round_id == 0:
            return 100
        if round_id == 1:
            return 0
        else:
            return credit_list[agent_id] * 0.01

    if type == 3:
        # init token + (Personal carbon credit*β)
        if round_id == 0:
            return 100
        else:
            return credit_list[agent_id] * 0.01


def main():
    if len(sys.argv) >= 2:
        json_path = sys.argv[1]
    else:
        json_path = os.path.join("config", "config.json")
    df: dict = utils.read_config(json_path)

    sim_type: int = df["sim_type"]
    baseline_type: int = df["baseline_type"]
    emission_type: int = df["emission_type"]
    coefficient_type: int = df["coefficient_type"]
    token_type: int = df["token_type"]

    # vis
    vis_token_alloc_plot = df["vis_token_alloc_plot"]
    vis_anim_token_alloc_plot = df["vis_anim_token_alloc_plot"]

    num_rounds: int = df["num_rounds"]
    num_agent: int = df["num_agent"]
    avg_window_size: int = df["avg_window_size"]  # rounds

    num_update_cycle: int = df["num_update_cycle"]
    # update cycle of carbon emission baseline

    carbon_emission_baseline = np.zeros([num_agent, num_rounds])
    for i in range(num_agent):
        carbon_emission_baseline[i] = get_baseline(
            df, i, num_agent, num_rounds, num_update_cycle, type=baseline_type
        )

    weekly_emission = np.zeros([num_agent, num_rounds])
    for i in range(num_agent):
        weekly_emission[i] = generate_emission(
            df, i, num_rounds, carbon_emission_baseline, type=emission_type
        )

    token = np.zeros((num_rounds, num_agent))
    for j in range(num_agent):
        token[0, j] += get_token_settlement(df, j, 0, type=token_type)
    print(f"Round #{0}")
    print(token)

    for i in range(1, num_rounds):

        credit_list = []
        score_list = []
        data = []

        # Calc Data
        for j in range(num_agent):
            personal_weekly_emission = weekly_emission[j, i]
            avg_personal_weekly_emission = np.mean(
                weekly_emission[j, max(i - avg_window_size, 0) : i + 1]
            )
            personal_weekly_reduction = (
                weekly_emission[j, i] - weekly_emission[j, i - 1]
            )
            compare_to_others = personal_weekly_emission - np.mean(
                weekly_emission[:, i]
            )

            data.append(
                [
                    avg_personal_weekly_emission,
                    personal_weekly_emission,
                    personal_weekly_reduction,
                    compare_to_others,
                ]
            )

        # Clac Coefficient
        coefficient = []
        for j in range(num_agent):
            a, b, c, d = get_coefficient(df, j, i, data[j], coefficient_type)
            coefficient.append([a, b, c, d])
        coefficient = np.mean(np.array(coefficient), axis=0)
        a, b, c, d = list(coefficient)

        # Clac Score and Credit
        score_list = np.sum(coefficient * data, axis=1)
        credit_list = carbon_emission_baseline[:, i] - score_list / 100

        # print(score_list)
        # print(credit_list)
        # print()

        # for j in range(num_agent):
        #     personal_score = (
        #         a * data[0]
        #         + b * personal_weekly_emission
        #         + c * personal_weekly_reduction
        #         + d * compare_to_others
        #     )
        #     personal_credit = carbon_emission_baseline[j, i] - personal_score / 100
        #     score_list.append(personal_score)
        #     credit_list.append(personal_credit)

        # Calc Token
        quota = np.sum(carbon_emission_baseline[:, i]) - np.sum(weekly_emission[:, i])
        for j in range(num_agent):
            token[i, j] = token[i - 1, j] + get_token_settlement(
                df, j, i, credit_list, score_list, quota, token_type
            )
        print(f"Round #{i}")
        print(token)

    print(np.sum(token, axis=1))

    if vis_token_alloc_plot:
        utils.plot_allocation(token[-1])
    if vis_anim_token_alloc_plot:
        utils.animate_plot_allocation(token)


if __name__ == "__main__":
    main()
