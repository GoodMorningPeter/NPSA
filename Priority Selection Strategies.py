import copy
import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

HIC_current = np.array(
    [99.1, 69.5, 88.0, 94.1, 76.6, 85.1, 72.5, 78.2, 94.5, 52.0, 90.6, 67.8, 45.1, 62.3, 56.0, 72.0, 63.1], dtype=float)
HIC_weights = {'centrality': 0.20, 'system_impact': 0.50, 'urgency': 0.20, 'modification': 0.10}
HIC_investment = np.array([0.19, 0.12, 0.11, 0.08, 0.20, 0.27, 0.09, 0.09, 0.07, 0.02], dtype=float)

UMIC_current = np.array(
    [98.3, 82.0, 82.6, 99.5, 77.1, 70.9, 63.8, 72.7, 77.5, 34.5, 79.5, 90.6, 85.5, 51.4, 49.2, 70.3, 44.8], dtype=float)
UMIC_weights = {'centrality': 0.20, 'system_impact': 0.35, 'urgency': 0.25, 'modification': 0.20}
UMIC_investment = np.array([1.99, 1.38, 3.75, 0.97, 3.38, 1.91, 0.55, 1.95, 2.20, 1.14], dtype=float)

LMIC_current = np.array(
    [56.4, 52.0, 62.3, 82.9, 33.9, 58.6, 68.7, 69.8, 44.2, 38.5, 50.3, 96.2, 96.2, 54.5, 50.6, 54.3, 50.2], dtype=float)
LMIC_weights = {'centrality': 0.20, 'system_impact': 0.25, 'urgency': 0.30, 'modification': 0.25}
LMIC_investment = np.array([1.27, 0.88, 0.20, 0.07, 0.10, 0.09, 0.05, 0.46, 0.53, 0.18], dtype=float)

LIC_current = np.array(
    [65.8, 62.6, 44.7, 37.2, 53.3, 38.7, 58.0, 60.9, 26.4, 71.7, 55.2, 97.4, 97.7, 66.1, 58.2, 42.2, 44.2], dtype=float)
LIC_weights = {'centrality': 0.20, 'system_impact': 0.15, 'urgency': 0.35, 'modification': 0.30}
LIC_investment = np.array([1.35, 1.15, 3.7, 1.1, 3.9, 0.75, 0.15, 0.87, 0.9, 0.1], dtype=float)


class INFLUENCE:
    def __init__(self, time: int, value: float):
        self.time = time
        self.value = value


class SDGNetwork:
    def __init__(self, initial_value: np.ndarray = None, target_value: np.ndarray = None, weights: dict = None):
        self.num_goals = 17
        self.num_resources = 10

        self.network = np.array([
            [0, -1, -1, -1, -1, -1, -1, -1, 1, 3, -1, -1, 1, 0, 0, 3, 2],
            [3, 0, 3, 2, 2, 0, -1, 1, 0, 3, -1, -1, -1, -1, 3, 0, 0],
            [3, 1, 0, 2, 2, 1, -1, 2, -1, 3, 1, 0, 0, 1, 0, 0, 0],
            [2, 2, 2, 0, 2, 2, 1, 2, 2, 3, 2, 2, 0, 2, 1, 2, 2],
            [3, 3, 2, 2, 0, -1, -1, 3, 0, 3, 2, -1, 0, 0, 2, 3, 1],
            [3, 3, 3, 2, 2, 0, 1, 3, 1, 3, 3, 1, 1, 2, 3, 0, 0],
            [3, 1, 3, 2, 0, 2, 0, 3, 3, 3, 3, -1, -1, -2, -2, 0, 2],
            [3, 2, 2, 3, 3, 2, 2, 0, 2, -1, 2, 3, -2, -2, -2, 0, 2],
            [2, 2, 2, 1, 0, 2, 3, 3, 0, -1, 3, -1, -1, 0, -2, 2, 2],
            [3, 2, 2, 0, 3, -2, -2, 3, 0, 0, -2, -2, -1, -1, -2, 0, 1],
            [1, 1, 2, 1, 3, 3, 2, 1, 3, 1, 0, 2, -1, 0, 1, 0, 1],
            [2, 2, 1, 2, 1, 1, 2, 3, 2, 3, 2, 0, -1, -1, -1, 0, 0],
            [2, 1, -1, 0, 1, -1, -2, -1, -1, 2, -2, -1, 0, 3, 2, 0, 0],
            [-1, 1, 1, 0, 0, 1, 1, -1, 1, -1, 1, 1, 0, 0, 2, 0, 0],
            [2, 3, 2, 0, 0, 2, -1, -1, -1, 0, 0, -1, 3, 3, 0, 0, 0],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 0, 3],
            [3, 2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2, 3, 3, 3, 3, 0]
        ])
        self.delay = np.array([
            [1, 3, 2, 5, 4, 3, 1, 4, 2, 3, 4, 2, 1, 2, 4, 3, 1],
            [1, 3, 2, 5, 4, 3, 4, 2, 1, 2, 4, 3, 1, 1, 4, 2, 3],
            [4, 2, 1, 3, 2, 5, 4, 2, 4, 3, 1, 3, 1, 3, 4, 2, 1],
            [1, 4, 2, 1, 4, 2, 1, 2, 3, 2, 5, 4, 3, 3, 4, 3, 1],
            [1, 3, 2, 5, 3, 4, 2, 1, 2, 4, 3, 1, 4, 3, 1, 4, 2],
            [4, 2, 2, 5, 1, 3, 4, 2, 3, 3, 1, 4, 3, 1, 1, 2, 4],
            [2, 5, 4, 2, 1, 1, 3, 4, 3, 1, 4, 2, 3, 2, 4, 3, 1],
            [1, 3, 4, 3, 1, 4, 2, 3, 4, 2, 1, 2, 4, 3, 1, 2, 5],
            [1, 3, 2, 5, 4, 3, 1, 4, 2, 3, 4, 2, 1, 2, 4, 3, 1],
            [1, 2, 3, 4, 2, 2, 4, 3, 3, 2, 5, 4, 1, 1, 4, 3, 1],
            [1, 3, 2, 5, 4, 3, 1, 4, 2, 3, 4, 2, 1, 2, 4, 3, 1],
            [3, 1, 4, 2, 1, 1, 3, 4, 2, 3, 2, 5, 4, 2, 4, 3, 1],
            [1, 3, 2, 5, 4, 3, 1, 4, 2, 3, 4, 2, 1, 2, 4, 3, 1],
            [1, 4, 3, 2, 3, 4, 2, 4, 1, 3, 2, 5, 1, 2, 4, 3, 1],
            [1, 3, 2, 4, 2, 1, 2, 4, 3, 1, 3, 4, 2, 1, 5, 4, 3],
            [4, 2, 4, 3, 1, 3, 4, 1, 3, 2, 5, 2, 1, 2, 4, 3, 1],
            [3, 1, 4, 2, 2, 5, 4, 3, 1, 3, 1, 2, 4, 3, 1, 4, 2]
        ])
        self.resource2goals = np.array([
            [0.05, 0.01, 0.03, 0.14, 0.40, 0.02, 0.06, 0.03, 0.05, 0.02, 0.04, 0.02, 0.01, 0.03, 0.01, 0.04, 0.04],
            [0.02, 0.01, 0.03, 0.01, 0.04, 0.04, 0.05, 0.01, 0.03, 0.14, 0.40, 0.02, 0.06, 0.03, 0.05, 0.02, 0.04],
            [0.02, 0.01, 0.03, 0.01, 0.04, 0.04, 0.40, 0.02, 0.06, 0.03, 0.05, 0.02, 0.04, 0.05, 0.01, 0.03, 0.14],
            [0.02, 0.06, 0.03, 0.05, 0.02, 0.04, 0.05, 0.01, 0.03, 0.14, 0.01, 0.03, 0.01, 0.04, 0.04, 0.40, 0.02],
            [0.05, 0.02, 0.03, 0.01, 0.04, 0.05, 0.01, 0.04, 0.40, 0.01, 0.04, 0.03, 0.14, 0.02, 0.02, 0.06, 0.03],
            [0.02, 0.06, 0.03, 0.05, 0.02, 0.04, 0.05, 0.01, 0.04, 0.04, 0.40, 0.03, 0.14, 0.01, 0.03, 0.01, 0.02],
            [0.01, 0.06, 0.04, 0.02, 0.04, 0.05, 0.04, 0.03, 0.14, 0.01, 0.03, 0.05, 0.02, 0.03, 0.01, 0.02, 0.40],
            [0.02, 0.06, 0.01, 0.05, 0.02, 0.04, 0.03, 0.04, 0.05, 0.04, 0.40, 0.01, 0.03, 0.03, 0.14, 0.01, 0.02],
            [0.03, 0.05, 0.02, 0.06, 0.03, 0.14, 0.01, 0.03, 0.01, 0.04, 0.05, 0.01, 0.02, 0.04, 0.04, 0.40, 0.02],
            [0.02, 0.04, 0.05, 0.02, 0.06, 0.03, 0.05, 0.01, 0.03, 0.14, 0.01, 0.03, 0.01, 0.04, 0.04, 0.40, 0.02]
        ])
        self.resource2goals = np.array([
            [6, 9, 3, 6, 6, 3, 1, 1, 0, 2, 1, 1, 1, 0, 1, 0, 6],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 1, 0, 0, 0, 0, 10, 0],
            [0, 0, 1, 6, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [1, 2, 3, 0, 0, 1, 4, 1, 5, 0, 1, 8, 3, 1, 2, 0, 8],
            [6, 11, 3, 2, 3, 3, 2, 12, 4, 9, 4, 1, 1, 1, 2, 0, 3],
            [0, 1, 9, 0, 0, 3, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 2],
            [1, 3, 3, 0, 0, 2, 3, 0, 0, 0, 0, 0, 3, 0, 9, 0, 0],
            [2, 2, 3, 7, 9, 3, 1, 1, 1, 7, 10, 0, 0, 0, 1, 0, 2],
            [0, 0, 1, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=float)
        for i in range(self.num_goals):
            self.resource2goals[:, i] = self.normalize(self.resource2goals[:, i])
        assert self.network.shape[0] == self.num_goals and self.network.shape[1] == self.num_goals
        assert self.delay.shape[0] == self.num_goals and self.delay.shape[1] == self.num_goals
        assert self.resource2goals.shape[0] == self.num_resources and self.resource2goals.shape[1] == self.num_goals

        self.sigmoid_gamma = 8
        self.simulate_time = 120

        self.target_value = target_value if target_value is not None else np.array(
            [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100], dtype=float)
        self.current_value = initial_value if initial_value is not None else np.array(
            [98.3, 82.0, 82.6, 99.5, 77.1, 70.9, 63.8, 72.7, 77.5, 34.5, 79.5, 90.6, 85.5, 51.4, 49.2, 70.3, 44.8],
            dtype=float)
        self.initial_value = copy.deepcopy(self.current_value)
        self.initial_progress = np.mean(self.current_value / self.target_value)
        self.influence_logger = [[] for _ in range(self.num_goals)]

        js_matrix = np.zeros([17, 17], dtype=float)
        for i in range(self.num_goals):
            for j in range(self.num_goals):
                if i < j:
                    js_matrix[i][j] = self.js_divergence(self.resource2goals[:, i], self.resource2goals[:, j])
        for i in range(self.num_goals):
            for j in range(self.num_goals):
                if i > j:
                    js_matrix[i][j] = js_matrix[j][i]
        self.system_impact = self.reversed_normalize(np.sum(js_matrix, axis=0))

        # Calculate IR & NI.
        self.influence = np.sum(self.network, axis=1)
        self.dependence = np.sum(self.network, axis=0)
        self.influence_ratio = self.normalize(self.influence / self.dependence)
        self.net_influence = self.normalize(self.influence - self.dependence)

        # Set urgency level for each goal.
        self.urgency = np.empty(17, dtype=float)
        self.modification = self.normalize(
            np.array([1, 4, 4, 4, 4, 4, 3, 3, 3, 1, 3, 3, 2, 2, 2, 4, 4], dtype=float))  # TODO

        # Set weights for different scores.
        if weights is not None:
            assert ['centrality', 'system_impact', 'urgency', 'modification'] == list(weights.keys())
        self.weights = weights if weights is not None else {'centrality': 0.2, 'system_impact': 0.2, 'urgency': 0.5,
                                                            'modification': 0.1}
        sum_weights = 0
        for key in self.weights:
            sum_weights += self.weights[key]
        assert math.fabs(sum_weights - 1.0) < 1e-8

    @staticmethod
    def normalize(array):
        _range = np.max(array) - np.min(array)
        return (array - np.min(array)) / _range

    @staticmethod
    def reversed_normalize(array):
        _range = np.max(array) - np.min(array)
        return (np.max(array) - array) / _range

    @staticmethod
    def js_divergence(p, q):
        m = (p + q) / 2
        return 0.5 * scipy.stats.entropy(p, m, base=2) + 0.5 * scipy.stats.entropy(q, m, base=2)

    def derivative_s_curve(self, x: float) -> float:
        return self.sigmoid_gamma * math.exp(self.sigmoid_gamma * (0.5 - x)) / math.pow(
            1 + math.exp(self.sigmoid_gamma * (0.5 - x)), 2)

    def update_urgency(self):
        progress = self.current_value / self.target_value
        self.urgency = 10 - progress // 0.1
        self.urgency = self.normalize(self.urgency)

    def calc_priority(self):
        self.update_urgency()
        priority_score = self.weights['centrality'] * (0.5 * self.influence_ratio + 0.5 * self.net_influence) + \
                         self.weights['system_impact'] * self.system_impact + \
                         self.weights['urgency'] * self.urgency + \
                         self.weights['modification'] * self.modification
        return priority_score

    def log_influence(self, delta):
        progress = self.current_value / self.target_value
        s_curve_derivatives = np.array([self.derivative_s_curve(x) for x in progress])
        new_influence = np.array([delta * s_curve_derivatives]).T * self.network * 1e-2  # TODO
        for src_goal in range(self.num_goals):
            for tgt_goal in range(self.num_goals):
                flag = 0
                for log in self.influence_logger[tgt_goal]:
                    if log.time == self.delay[src_goal][tgt_goal]:
                        log.value += new_influence[src_goal][tgt_goal]
                        flag = 1
                        break
                if flag == 0:
                    self.influence_logger[tgt_goal].append(
                        INFLUENCE(time=self.delay[src_goal][tgt_goal], value=new_influence[src_goal][tgt_goal]))

    def update_influence_logger_time(self):
        for goal_no in range(self.num_goals):
            zero_index = -1
            for log_no in range(len(self.influence_logger[goal_no])):
                self.influence_logger[goal_no][log_no].time -= 1
                if self.influence_logger[goal_no][log_no].time == 0:
                    zero_index = log_no
            if zero_index != -1:
                self.influence_logger[goal_no].pop(zero_index)

    def propagate_single(self, investment):
        # Calculate the contribution of direct investment.
        temp_delta = investment * 0.1  # TODO

        # Calculate the contribution of other goals' influence in history.
        for i in range(self.num_goals):
            for item in self.influence_logger[i]:
                if item.time == 1:
                    temp_delta[i] += item.value
                    break

        # Update the time in the influence logger.
        self.update_influence_logger_time()

        # Add new influence to the logger.
        self.log_influence(delta=temp_delta)

        # Update the current value of each goal.
        self.current_value += temp_delta

    def propagate(self, investment: np.ndarray):
        investment = self.normalize(np.sum(np.array([investment]).T * self.resource2goals, axis=0))
        print(f'Year 1 rank: {1 + np.argsort(-investment)}')
        for i in range(self.simulate_time):
            if i % 12 == 0:
                priority_scores = self.calc_priority()
                print(f'Year {i // 12 + 1} rank: {1 + np.argsort(-priority_scores)}')
                priority_scores[13:] = 0
                investment = self.normalize(priority_scores)
            self.propagate_single(investment)

    def radar_display(self, progress):
        results = {f'SDG {i + 1}': 100 * progress[i] for i in range(self.num_goals)}
        data_length = len(results)
        angles = np.linspace(0, 2 * np.pi, data_length, endpoint=False)
        labels = list(results.keys())
        score = list(results.values())
        score = np.concatenate((score, [score[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        labels = np.concatenate((labels, [labels[0]]))

        plt.rc('font', family='Times New Roman')
        fig = plt.figure(figsize=(10, 6), dpi=300)
        fig.suptitle('')
        ax = plt.subplot(121, polar=True)
        for j in np.arange(0, 100 + 20, 20):
            ax.plot(angles, (self.num_goals + 1) * [j], '-.', lw=0.5, color='black')
        for j in range(self.num_goals):
            ax.plot([angles[j], angles[j]], [0, 120], '-.', lw=0.5, color='black')
        ax.plot(angles, score, color='b')
        ax.spines['polar'].set_visible(False)
        ax.grid(False)
        for a, b in zip(angles, score):
            ax.text(a, b + 5, '%.00f' % b, ha='center', va='center', fontsize=12, color='b')
        ax.set_thetagrids(angles * 180 / np.pi, labels)
        ax.set_theta_zero_location('N')
        ax.set_rlim(0, 110)
        ax.set_rlabel_position(0)
        ax.set_title('')
        plt.show()

    def summary(self):
        print(f'Progress of each goal in {self.simulate_time} months:')
        for i in range(self.num_goals):
            if self.current_value[i] > 100.0:
                self.current_value[i] = 100.0
        final = self.current_value / self.target_value
        initial = self.initial_value / self.target_value
        delta = final - initial
        final = initial + 0.92 * delta
        self.radar_display(final)
        for i in range(self.num_goals):
            print(f'    SDG {i + 1} : {round(100 * initial[i], 2)} - {round(100 * final[i], 2)} '
                  f'({round(100 * delta[i], 2)})')
            print(f'{round(100 * final[i], 2)}')
        print(f'Average progress: {round(100 * self.initial_progress, 2)} - {round(100 * np.mean(final), 2)} '
              f'({round(100 * (np.mean(final) - self.initial_progress), 2)})%')
        print(f'Standard deviation: {np.std(100 * final)}')
        return final


if __name__ == '__main__':
    sdg_net = SDGNetwork(
        initial_value=HIC_current,
        weights=HIC_weights
    )
    sdg_net.propagate(investment=HIC_investment)
    final_ = sdg_net.summary()
