import multiprocessing
import random, itertools, statistics, collections
from functools import lru_cache, partial
import argparse
import csv
from multiprocessing.dummy import freeze_support

import scipy.stats as stats
import numpy as np

DEBUG = False # Debug flag
"""
    Represents a landscape (i.e. circle) on which the algorithm runs
    
    (Internally this is just a list with a fancy method to get values at different positions)
"""
class Landscape:
    def __init__(self, N, max = 100):
        self.values = []
        self.N = N
        for i in range(N):
            self.values.append( random.uniform(0, max))
    @lru_cache(maxsize=400)
    def valueAt(self, i) -> int:
        """
        Gets the value at the (i mod N)-th position on the landscape.
        :param i: Index
        :return: Value at the position specified by i

        Since this method is called often, it is cached with an lru-cache
        """
        return self.values[i % self.N]



class Agent:
    """
    Represents an agent.
    Holds the agent's strategy as a list of numbers.
    """
    def __init__(self, strategy):
        self.strategy = strategy
        self.singleScore = None

    def nextMove(self, landscape : Landscape, position):
        """
        Computes where the agent should move next
        :param landscape: landscape the agent should move on
        :param position: the current position of the agent
        :return: the agent's next position based on her strategy
        """
        current = landscape.valueAt(position)
        for offset in self.strategy:
            # If the value is higher, then where we are currently standing, we will try to move to that position
            if landscape.valueAt(position + offset) > current:
                return (position + offset) % (landscape.N)
        return position


class HongPageSimulation:
    """
    This implements the classic Hong & Page simulation.
    The parameter m is the maximum lookahead an agent can have. In the original paper, m = 12.

    Each of the other types of teamwork inherit from this class and override the simulateMultirun methods.
    Some also additionally override the simulateSingle method.

    """


    def __init__(self, m = 12, N = 2000):
        self.landscape = Landscape(N = N)
        self.agents = []
        # Generate all the agents
        # the length of the permutation is 3
        for x in itertools.permutations(range(1, m+1), 3):
            self.agents.append(Agent(x))
        if DEBUG:
            print("Initialized %d agents."%(len(self.agents)))

    def run(self):
        """
        Simulates two teams, one with the best agents, one with random agents
        :return: a quadruple containing score of team of best agents and the team of random agents, and their diversities
        """
        # Evaluate each agent from each starting position individually
        for agent in self.agents:
            agent.singleScore = self.scoreAgent(agent)
            if DEBUG:
                print(agent.strategy, agent.singleScore)
        # Sort agents by single score
        sortedAgents = sorted(self.agents, key = lambda a: a.singleScore, reverse=True)
        # Just take the 10 best agents for the first team
        teamOfBest = sortedAgents[0:9]
        # the random team is sampled uniformely
        randomTeam = random.sample(self.agents, 10)
        # Compute scores
        scoreBest = (self.scoreTeam(teamOfBest))
        scoreRand = (self.scoreTeam(randomTeam))
        # Calculate the teams diversity
        divBest = self.calculateDiversity(teamOfBest)
        divRnd = self.calculateDiversity(randomTeam)
        return (scoreBest, scoreRand, divBest, divRnd)

    def calculateDiversity(self, team):
        """
        Calulates the diversity of a team.
        (Inspired by Wallrich's implementation)
        """
        vals = []
        for i in range(len(team)):
            for j in range(i + 1, len(team)):
                res = (len(team[i].strategy) - sum(x == y for x, y in zip(team[i].strategy, team[j].strategy))) / len(team[i].strategy)
                vals.append(res)
        return statistics.mean(vals)


    def scoreAgent(self, agent):
        """
        Computes an individual agent's (average) score on the landscape
        :param agent: the agent to simulate
        :return: the average of his score, starting at each position of the landscpae
        """
        sum = 0
        for pos in range(self.landscape.N):
            sum += self.simulateSingle(agent, pos)
        return (sum / self.landscape.N)

    def simulateSingle(self, agent, start):
        """
        This method simulates a single agent starting at a specific position
        and running until it is no longer possible to improve.
        :param agent: The agent to simulate
        :param start: starting index
        :return: the (local) maximum the agent has found
        """
        pos = start
        i = 0
        while True:
            nextpos = agent.nextMove(self.landscape, pos)
            if nextpos == pos:
                return self.landscape.valueAt(pos)
            pos = nextpos
            i = i+1


    def scoreTeam(self, team):
        """
        The score of a team is computed by letting it start from every position and computing the average of their scores
        :param team: list of agents
        :return: average score when starting from each position
        """
        sum = 0
        for pos in range(self.landscape.N):
            sum += self.simulateMultirun(team, pos)
        return (sum / self.landscape.N)

    def simulateMultirun(self, team, start):
        """
        Simulates a team of agents collaborating, using the standard method.
        :param team: a list with the agents compromising the team
        :param start: starting position
        :return: the score the team has reached collectively
        """
        current = start
        currentAgentIndex = 0

        currentAgent = team[currentAgentIndex]
        # SkipCount is used to check,
        skipCount = 0
        while True:
            nextpos = currentAgent.nextMove(self.landscape, current)
            if nextpos == current:
                skipCount += 1
                currentAgentIndex = (currentAgentIndex + 1) % (len(team))
                currentAgent = team[currentAgentIndex]
                if skipCount > len(team):
                    return self.landscape.valueAt(current)
                continue
            skipCount = 0
            current = nextpos



    def getName(self):
        """
        Method used for getting the name of the kind of teamwork that is done
        """
        return "Hong-Page (default)"

class TournamentSimulation(HongPageSimulation):
    """"
        In a tournament, every agent's lookahead is considered
    """

    def simulateMultirun(self, team, start):
        current = start
        while True:
            old = current
            # Find the best spot, by consulting every agent
            for agent in team:
                nextpos = agent.nextMove(self.landscape, old)
                # Does this agent's move lead to a better position?
                if self.landscape.valueAt(nextpos) > self.landscape.valueAt(current):
                    current = nextpos
            if current == old:
                return self.landscape.valueAt(current)
    def getName(self):
        return "Tournament"

class DemocraticSimulation(HongPageSimulation):
    """
    Uses a voting procedure (plurality vote) among the agents to select the next position
    """
    def simulateMultirun(self, team, start):
        current = start
        while True:
            votes = []
            # Get the next move from each agent
            for agent in team:
                nextpos = agent.nextMove(self.landscape, current)
                if nextpos != current:
                    votes.append(nextpos)
            if not votes: # If no agent has a better position -> stop
                return self.landscape.valueAt(current)
            # Get majority element
            counter = collections.Counter(votes)
            # Use the most common element and move there
            current = counter.most_common()[0][0]

    def getName(self):
        return "Democracy"


class ChancyError(HongPageSimulation):
    """
    Implements a relay scheme, where with a certain probability the agent just moves to the offset indicated by the
    first number in their strategy vector.
    """
    def simulateSingle(self, agent, start):
        pos = start
        i = 0
        while True:
            # Throw the dice
            if random.uniform(0,1) < 0.05: # 5% chance
                nextpos = pos + agent.strategy[1]
            else:
                nextpos = agent.nextMove(self.landscape, pos)
            if nextpos == pos:
                return self.landscape.valueAt(pos)
            pos = nextpos
            i = i+1

    def getName(self):
        return "Non-monotonous relay"

class RandomDictator(HongPageSimulation):
    def simulateMultirun(self, team, start):
        current = start
        while True:
            # Choose a random agent, to be the dictator
            dictator = random.choice(team)
            # Compute with this agent
            next = dictator.nextMove(self.landscape, current)
            if next == current:
                return self.landscape.valueAt(current)
            current = next

    def getName(self):
        return "Random Dictator"


class PairRelay(HongPageSimulation):
    def simulateMultirun(self, team, start):
        # Choose two agents and let them cooperate
        current = start
        while True:
            # Choose two agents (different)
            a1 = random.choice(team)
            a2 = a1
            while a2 == a1:
                a2 = random.choice(team)
            # Let them cooperate by utilizing both agents powers
            lookahead = []
            for a in a1.strategy:
                lookahead.append(a)
                for b in a2.strategy:
                    lookahead.append(b)
                    lookahead.append(a+b)
            # De-duplicate (for efficency)
            lookahead = list(set(lookahead))
            old = current
            # Check which ofset from the ones in lookahead is best
            for x in lookahead:
                if self.landscape.valueAt(current + x ) > self.landscape.valueAt(current):
                    current = (current + x) % (self.landscape.N)
            if old == current:
                return self.landscape.valueAt(current)
    def getName(self):
        return "Pair Relay"


class SimplePairRelay(HongPageSimulation):
    def simulateMultirun(self, team, start):
        # Choose two agents and let them cooperate
        current = start
        while True:
            # Choose two agents (different)
            a1 = random.choice(team)
            a2 = a1
            while a2 == a1:
                a2 = random.choice(team)
            # Let them cooperate by utilizing both agents powers
            lookahead = a1.strategy + a2.strategy
            # De-duplicate
            lookahead = list(set(lookahead))
            old = current
            for x in lookahead:
                if self.landscape.valueAt(current + x ) > self.landscape.valueAt(current):
                    current = (current + x) % (self.landscape.N)
            if old == current:
                return self.landscape.valueAt(current)
    def getName(self):
        return "Simple Pair Relay"




class BadTeamWork(HongPageSimulation):
    def simulateMultirun(self, team, start):
        max = start
        # Simulates each agent individually
        for agent in team:
            cur = start
            while True:
                np = agent.nextMove(self.landscape, cur)
                if np == cur:
                    break
                cur = np
            if self.landscape.valueAt(cur) > self.landscape.valueAt(max):
                max = cur
        return self.landscape.valueAt(max)

    def getName(self):
        return "Bad Teamwork"


def evaluate(modelClass, N, M):
    """
    Evaluates the given class
    :param modelClass: the type of teamwork to use (given as a class)
    :param N: size of the landscape
    :param M: number of iterations
    :return: an array, that is later printed into the csv file
    """
    c1 = 0
    randomvalues = []
    bestvalues = []
    diversityRnd = []
    diversityBest = []
    for i in range(M):
        print(i)
        model = modelClass(N = N)
        r = model.run()
        #print(r)
        bestvalues.append(r[0])
        randomvalues.append(r[1])
        diversityBest.append(r[2])
        diversityRnd.append(r[3])
        if r[0] < r[1]:
            c1 = c1+1

        # Evaluate statistical significance
    pval = stats.ttest_ind(np.array(bestvalues), np.array(randomvalues), equal_var = False).pvalue
    print("significant" if pval < 0.05 else "insignificant")
    return [model.getName(), round(statistics.mean(randomvalues), 2), round(statistics.stdev(randomvalues), 2),
                     round(statistics.mean(bestvalues), 2),  round(statistics.stdev(bestvalues), 2),
                     round(statistics.mean(diversityRnd), 2),  round(statistics.stdev(diversityRnd), 2),
                     round(statistics.mean(diversityBest), 2),  round(statistics.stdev(diversityBest), 2),
                     pval
                     ]

def main():
    """
    Main method
    """
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Run a Hong & Page style simulation.')
    parser.add_argument('-o', metavar='file', dest="file",
                        default='output.csv', type=argparse.FileType('w'),
                        help='file to write results to (defaults to output.csv)')
    parser.add_argument('-M', metavar='M',
                        default=50, type=int,
                        help='number of iterations per strategy (default 50)')

    parser.add_argument('-N', metavar='N',
                        default=2000, type=int,
                        help='size of landscape (default 2000)')
    parser.add_argument('-t', metavar='t',
                        default=8, type=int,
                        help='number of threads (default 8)')
    args = parser.parse_args()
    # Create a list of the teamworks we consider
    teamworks = [HongPageSimulation, TournamentSimulation, DemocraticSimulation, ChancyError, RandomDictator, PairRelay, SimplePairRelay, BadTeamWork]
    # Create a pool with t threads, for multithreading
    pool = multiprocessing.Pool(args.t)
    # Get the results as a list of outputs
    results = pool.map(partial(evaluate, N=args.N, M=args.M), teamworks)
    pool.close()
    pool.join()

    print(results)
    # Write the results to the file
    writer = csv.writer(args.file)
    writer.writerow(["Model", "Avg random", "Stdev random", "Avg best", "Stdev best", "Diversity Random", "Stdev Diversity Random", "Diversity Best", "Stdev Diversity Best", "pval"])
    writer.writerows(results)

if __name__ == '__main__':
    freeze_support()
    main()
