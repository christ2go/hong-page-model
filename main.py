import random, itertools, statistics, collections
from functools import lru_cache

DEBUG = False # Debug flag
"""
    Represents a landscape (i.e. circle) on which the algorithm runs
    
    (Internally this is just a list with a fancy method to get values at different positions)
"""
class Landscape:
    def __init__(self, N = 2000, max = 100):
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
        """
        return self.values[i % self.N]



"""
    Represents an agent. 
    Holds the agent's strategy as a list of numbers.
"""
class Agent:
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


"""
    This implements the classic Hong & Page simulation.  
    The parameter m is the maximum lookahead an agent can have. In the original paper, m = 12.
"""
class HongPageSimulation:
    def __init__(self, m = 12):
        self.landscape = Landscape()
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
        :return: a tuple containing score of team of best agents and the team of random agents, in that order
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
        scoreDiv = (self.scoreTeam(randomTeam))
        return (scoreBest, scoreDiv)

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


    """
        Outputs the data as a .csv file, for generating tables, etc 
    """
    def archive(self):
        pass

    def getName(self):
        return "Hong-Page (default)"

""""
    In a tournament, every agent's llokahead is considered
"""
class TournamentSimulation(HongPageSimulation):

    def simulateMultirun(self, team, start):
        current = start
        while True:
            old = current
            for agent in team:
                nextpos = agent.nextMove(self.landscape, old)
                if self.landscape.valueAt(nextpos) > self.landscape.valueAt(current):
                    current = nextpos
            if current == old:
                return self.landscape.valueAt(current)
    def getName(self):
        return "Tournament"

class DemocraticSimulation(HongPageSimulation):
    """
    Uses a voting procedure among the agents to select the next position
    """
    def simulateMultirun(self, team, start):
        current = start
        while True:
            votes = []
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
    def simulateSingle(self, agent, start):
        pos = start
        i = 0
        while True:
            if random.uniform(0,1) < 0.05:
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
            # Choose a random agent
            agent = random.choice(team)
            # Compute with this agent
            next = agent.nextMove(self.landscape, current)
            if next == current:
                return self.landscape.valueAt(current)
            current = next

    def getName(self):
        return "Random Dictator"

class PairRelay(HongPageSimulation):
    def simulateMultirun(self, team, start):
        # Choose two agents and let them cooperate
        pass

class BadTeamWork(HongPageSimulation):
    pass

def evaluate(modelClass):
    c1 = 0
    randomvalues = []
    bestvalues = []
    for i in range(500):
        model = modelClass()
        r = model.run()
        #print(r)
        randomvalues.append(r[1])
        bestvalues.append(r[0])
        if r[0] < r[1]:
            c1 = c1+1

    print(model.getName())

    print("%.2f (%.2f)"%((statistics.mean(randomvalues), statistics.stdev(randomvalues))))

    print("%.2f (%.2f)"%((statistics.mean(bestvalues), statistics.stdev(bestvalues))))
teamworks = [HongPageSimulation, TournamentSimulation, RandomDictator, ChancyError, DemocraticSimulation]
for x in teamworks:
    evaluate(x)

#### Export results into a .csv