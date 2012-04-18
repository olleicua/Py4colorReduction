"""
TODO: add better docs here

added test result file.  Generate it with:

 python reductionAttempt.py > testResults.txt

"""

import itertools, copy, sys

# UTILS #

# from http://stackoverflow.com/questions/2931672/what-is-the-cleanest-way-to-do-a-sort-plus-uniq-on-a-python-list
def sort_uniq(sequence):
	"Return a generator of the given values sorted sans duplicates."
	return (x[0] for x in itertools.groupby(sorted(sequence)))

# CONSTANTS #

COLORS = ["R", "B", "G", "Y"]
def colorToCSSColor(colorID) :
	if colorID == None: return None
	elif colorID == "R": return "#ff0000"
	elif colorID == "G": return "#00ff00"
	elif colorID == "B": return "#0000ff"
	elif colorID == "Y": return "#ffff00"
	else: raise ValueError("Invalid color")

# CLASSES #

class GraphFailedPreconditionError(Exception) :
	def __init__(self, value) :
		self.value = value
		#
	def __str__(self) :
		return repr(self.value)
	#
	def __cmp__(self, other) :
		return cmp(self.name, other.name)

class Node :
	"""
	Use:
	>>> Node("a")
	a_5
	>>> Node("b", 7)
	b_7
	
	Removed limit on minimum degree for reduction tests below now this fails
	
	- >>> Node("c", 0)
	Traceback (most recent call last):
	    ...
	ValueError: degree must be at least 5
	"""
	def __init__(self, name, degree=5) :
		self.name = name
		#if degree < 5 :
		#	raise ValueError("degree must be at least 5")
		self.degree = degree
		self.color = None
		#
	def __repr__(self) :
		return "%s_%d" % (self.name, self.degree)
	#
	def __cmp__(self, other) :
		return cmp(self.name, other.name)

class BoundaryNode :
	"""
	A node with unspecified degree for the boundary ring of a configuration.
	Use:
	->>> BoundaryNode()
	b0
	->>> BoundaryNode()
	b1
	"""
	count = 0
	def __init__(self) :
		self.name = "b%d|" % self.__class__.count
		self.__class__.count += 1
		self.color = None
		self.colorsTried = []
		#
	def __repr__(self) :
		return self.name
	
class Configuration :
	"""
	The initial configurations should be already triangulated as we have no
	 clearly defined algorithm to triangulate them.
	"""
	def __init__(self, nodes, adjacencyList) :
		"""
		nodes is a list of Node objects
		adjacencyList is a list of pairs (sorted tuples) of nodes names
		Use:
		>>> c = Configuration([Node("a"), Node("b", 7), Node("c", 8)],
		...                   [("a", "b"), ("a", "c")])
		>>> c.nodes["a"]
		a_5
		>>> c.adjacencyList
		[(a_5, b_7), (a_5, c_8)]
		"""
		# create a dictionary mapping node names to nodes
		self.nodes = {}
		for node in nodes :
			self.nodes[node.name] = node

		# build adjacency list using nodes instead of node names
		self.adjacencyList = []
		for nodeA, nodeB in adjacencyList :
			self.addEdge(self.nodes[nodeA], self.nodes[nodeB])
		#
	def __getitem__(self, name) :
		return self.nodes[name]
	#
	def addEdge(self, nodeA, nodeB) :
		"""
		Use:
		>>> config = Configuration([Node("a"), Node("b", 7)], [])
		>>> config.addEdge(config["a"], config["b"])
		>>> config.adjacencyList
		[(a_5, b_7)]
		>>> config.addEdge(config["a"], config["b"])
		>>> config.adjacencyList
		[(a_5, b_7)]
		"""
		if not tuple(sorted((nodeA, nodeB))) in self.adjacencyList :
			self.adjacencyList.append(tuple(sorted((nodeA, nodeB))))
		#
	def isEdge(self, nodeA, nodeB) :
		"""
		Use:
		>>> config = Configuration([Node("a"), Node("b"), Node("c")], [])
		>>> config.addEdge(config["a"], config["b"])
		>>> config.isEdge(config["b"], config["a"])
		True
		>>> config.isEdge(config["b"], config["c"])
		False
		"""
		return tuple(sorted((nodeA, nodeB))) in self.adjacencyList
	#
	def mergeNodes(self, mergeNodeA, mergeNodeB) :
		"""
		Merges node B into node A.
		>>> config = Configuration([Node("a"), Node("b", 7)],
		...                        [("a", "b")])
		>>> config.mergeNodes(config.nodes["a"], config.nodes["b"])
		>>> config.nodes
		{'a': a_5}
		>>> config.adjacencyList
		[]
		>>> config = Configuration([Node("a"), Node("b"), Node("c"), Node("d")],
		...                        [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")])
		>>> config.mergeNodes(config.nodes["b"], config.nodes["a"])
		>>> sorted(config.nodes.items())
		[('b', b_5), ('c', c_5), ('d', d_5)]
		>>> sorted(config.adjacencyList)
		[(b_5, c_5), (b_5, d_5), (c_5, d_5)]
		"""
		def toA(node) :
			if node == mergeNodeB :
				return mergeNodeA
			else :
				return node
		renamedAdjacencies = [tuple(sorted((toA(a),toA(b)))) for a, b in self.adjacencyList]
		# Parallel edges are forbidden:
		renamedUniqedAdjacencies = list(sort_uniq(renamedAdjacencies))
		# Loops are forbidden (namely, if there was an edge between A and B) :
		self.adjacencyList = filter(lambda (a,b): a != b, renamedUniqedAdjacencies)
		del self.nodes[mergeNodeB.name]
	#
	def getNeighbors(self, node) :
		"""
		Use:
		>>> config = Configuration([Node("a"), Node("b", 7), Node("c", 8)],
		...                        [("a", "b"), ("a", "c")])
		>>> config.getNeighbors(config["a"])
		[b_7, c_8]
		"""
		result = []
		for nodeA, nodeB in self.adjacencyList :
			if node == nodeA :
				result.append(nodeB)
			if node == nodeB :
				result.append(nodeA)
		return result
	#
	def getBoundaryNeighbors(self, node) :
		"""
		Use:
		>>> config = Configuration([Node("a"), Node("b", 7)],
		...                        [("a", "b")])
		>>> bn = BoundaryNode()
		>>> config.nodes[bn.name] = bn
		>>> config.addEdge(config["a"], bn)
		>>> config.getBoundaryNeighbors(config["a"]) == [bn]
		True
		"""
		result = []
		for nodeA, nodeB in self.adjacencyList :
			if node == nodeA and isinstance(nodeB, BoundaryNode) :
				result.append(nodeB)
			if node == nodeB and isinstance(nodeA, BoundaryNode) :
				result.append(nodeA)
		return result
	#
	def apparentDegree(self, node) :
		"""
		Use:
		>>> config = Configuration([Node("a"), Node("b", 7), Node("c", 8)],
		...                        [("a", "b"), ("a", "c")])
		>>> config.apparentDegree(config["a"])
		2
		"""
		return len(self.getNeighbors(node))
	#
	def isOuter(self, node) :
		"""
		Use:
		>>> config = Configuration([Node("a"), Node("b"), Node("c"), Node("d"),
		...                         Node("e"), Node("f")],
		...                        [("a", "b"), ("a", "c"), ("a", "d"),
		...                         ("a", "e"), ("a", "f")])
		>>> config.isOuter(config["a"])
		False
		>>> config.isOuter(config["b"])
		True
		"""
		if isinstance(node, BoundaryNode) :
			return False
		return node.degree > len(self.getNeighbors(node))
	#
	def allowedColors(self, node) :
		"""
		>>> config = Configuration([Node("a"), Node("b")],
		...                        [("a", "b")])
		>>> config["a"].color = "R"
		>>> sorted(config.allowedColors(config["b"]))
		['B', 'G', 'Y']
		"""
		return set(COLORS) - set(map(lambda node: node.color,
		                         self.getNeighbors(node)))
	def outerNodes(self) :
		"""
		Return a list of nodes that have currently unspecified edges.
		Use:
		>>> config = Configuration([Node("a"), Node("b"), Node("c"), Node("d"),
		...                         Node("e"), Node("f")],
		...                        [("a", "b"), ("a", "c"), ("a", "d"),
		...                         ("a", "e"), ("a", "f")])
		>>> sorted(config.outerNodes())
		[b_5, c_5, d_5, e_5, f_5]
		"""
		result = []
		for node in sorted(self.nodes.values()) :
			if self.isOuter(node) :
				result.append(node)
		return result
	#	
	def outerNodeCycle(self) :
		"""
		Return, roughly, the cycle of nodes that form the edge of the
		configuration (not the boundary nodes; the "outer" ones).
				
		Precisely, returns a list of nodes that form a minimal closed
		walk that traverses each outerNode at least once and traverses
		no non-outer nodes.
		
		The result is undefined when no such walk exists; e.g. a large
		tyre made up of a thick layer of nodes.  Such graphs would
		have to have two (or more) cycles to count all edge nodes.
		
		In this list, the first node is not duplicated as the last
		node.
		
		>>> config = Configuration([Node("a")], [])
		>>> config.outerNodeCycle()
		[a_5]
		
		>>> config = Configuration([Node("a"), Node("b")],
		...                        [("a", "b")])
		>>> config.outerNodeCycle()
		[a_5, b_5]
		
		>>> config = Configuration(
		...                        # A pentagon with centre node.
		...                        [Node("a"), Node("b"), Node("c"), Node("d"),
		...                         Node("e"), Node("f")],
		...                        [("a", "b"), ("a", "c"), ("a", "d"),
		...                         ("a", "e"), ("a", "f"),
		...                         ("b", "c"), ("c", "d"), ("d", "e"),
		...                         ("e", "f"), ("f", "b")])
		>>> config.outerNodeCycle()
		[b_5, c_5, d_5, e_5, f_5]
		
		The length of the list is the sum, over each outer node N, of
		the number of connected components the graph of "outer nodes"
		would contain if node N were to be removed.  In other words,
		chokepoints ("cutvertices") have to be traversed multiple times
		in order to get to all parts of the graph.  (My intuition
		claims this, but I have not proved this. --Isaac)
		
		TODO: The present algorithm is broken for configurations that
		have a cutvertex:

		>>> config = Configuration(
		...                        # Two triangles that share a node "a".
		...                        [Node("a",6), Node("b"), Node("c"),
		...                         Node("d"), Node("e")],
		...                        [("a", "b"), ("a", "c"), ("b", "c"),
		...                         ("a", "d"), ("a", "e"), ("d", "e")])
		>>> config.outerNodeCycle()
		[a_6, b_5, c_5, a_6, d_5, e_5]

		TODO: The present algorithm is factorial time.  This problem
		is more-or-less to find a Hamiltonian cycle amongst outer
		nodes.  Finding Hamiltonian cycles is NP-complete.  There's
		surely an exponential-time algorithm though (which is better
		than factorial time).
		"""
		#
		outerNodes = self.outerNodes()
		#
		# special-case for len(outerNodes) == 1
		# because a node does not have an edge going to itself,
		# and checking here is faster than checking in consistent().
		if len(outerNodes) <= 1 :
			return outerNodes
		#
		def consistent(ordering) :
			"""
			Returns True iff all edges in the ordering exist,
			including last---first.
			"""
			for i, nodeA in enumerate(ordering) :
				# ordering[-1] if i == 0: last node
				nodeB = ordering[i - 1]
				if not self.isEdge(nodeA, nodeB) :
					return False
			return True
		#
		for ordering in itertools.permutations(outerNodes) :
			if consistent(ordering) :
				return list(ordering)
		raise GraphFailedPreconditionError("no outer node cycle found")
		#
	def addBoundary(self) :
		"""
		Add boundary nodes (assumes that the configuartion is triangulated).
		 In cases like:
		     (5) -- (7) -- (5)
		 the behavior of this function will be undefined.
		Use:
		>>> config = Configuration([Node("a")], [])
		>>> config.addBoundary()
		>>> len(config.getBoundaryNeighbors(config["a"])) == 5
		True
		>>> sorted(config.adjacencyList)
		[(a_5, b0|), (a_5, b1|), (a_5, b2|), (a_5, b3|), (a_5, b4|), (b0|, b1|), (b0|, b4|), (b1|, b2|), (b2|, b3|), (b3|, b4|)]
		
		>>> config = Configuration([Node("a"), Node("b"), Node("c")],
		...                        [("a", "b"), ("a", "c"), ("c", "b")])
		>>> config.addBoundary()
		>>> len(config.getBoundaryNeighbors(config["a"])) == 3
		True
		>>> len(config.getBoundaryNeighbors(config["b"])) == 3
		True
		>>> len(config.getBoundaryNeighbors(config["c"])) == 3
		True
		>>> config = Configuration([Node("a", 6), Node("b")], [("a", "b")])
		>>> config.addBoundary()
		>>> len(config.nodes)
		9
		>>> len(config.adjacencyList)
		17
		>>> config.addBoundary()
		>>> len(config.adjacencyList)
		17
		"""
		# add triangulating boundary nodes between each pair in the outer cycle
		outerCycle = self.outerNodeCycle()
		for cycleI, node in enumerate(outerCycle) :
			newBoundaryNode = BoundaryNode()
			self.nodes[newBoundaryNode.name] = newBoundaryNode
			self.addEdge(newBoundaryNode, node)
			self.addEdge(newBoundaryNode, outerCycle[cycleI - 1])
			# cycleI - 1 wraps when cycleI == 0
			#
		# add remaining boundary nodes for each outer node
		for node in outerCycle :
			#
			# determine how many new nodes are needed if any
			newCount = node.degree - self.apparentDegree(node)
			assert newCount >= -1, "Node degree too high"
			# get previously triangulated BoundaryNodes
			triangleNodes = self.getBoundaryNeighbors(node)
			assert 0 < len(triangleNodes) <= 2, "Wrong number of BoundaryNodes"
			#
			# the below conditionals assume the configuration is two-connected
			if -1 == newCount :
				# more edges than degree : merge nodes
				assert 2 == len(triangleNodes), 'Needing to "merge" 1 node'
				self.mergeNodes(*triangleNodes)
				#
			elif 0 == newCount and 2 == len(triangleNodes) :
				 # degree satisfied join triangleNodes
				self.addEdge(*triangleNodes)
				#
			else :
				# generate new BoundaryNodes
				newBoundaryNodes = [BoundaryNode() for n in range(newCount)]
				#
				# connect new BoundaryNodes to previous triangulating ones
				triangleNodes = self.getBoundaryNeighbors(node)
				assert 0 < len(triangleNodes) <= 2, \
					"Too many or too few BoundaryNodes"
				self.addEdge(newBoundaryNodes[0], triangleNodes[0])
				self.addEdge(newBoundaryNodes[-1], triangleNodes[-1])
				# -1 cleanly handles case of exactly 1 triangleNode
				#
				# connect new BoundaryNodes to each other and this Node
				for newBNi, boundaryNode in enumerate(newBoundaryNodes) :
					self.nodes[boundaryNode.name] = boundaryNode
					self.addEdge(boundaryNode, node)
					if newBNi > 0 :
						self.addEdge(boundaryNode, newBoundaryNodes[newBNi - 1])
		#
	def getBoundaryCycle(self) :
		"""
		Returns a list of the boundary nodes in cyclic order.
		Automatically adds a boundary if it isn't there already.
		>>> config = Configuration([Node("a"), Node("b"), Node("c"), Node("d")],
		...                        [("a", "b"), ("b", "c"), ("c", "d"),
		...                         ("d", "a"), ("b", "d")])
		>>> len(config.getBoundaryCycle())
		6
		>>> len([pair for pair in config.adjacencyList
		...        if isinstance(pair[0], BoundaryNode) and isinstance(pair[1], BoundaryNode)])
		6
		"""
		self.addBoundary()
		unorderedBoundaryNodes = [node for node in self.nodes.values() if
						isinstance(node, BoundaryNode)]
		if len(unorderedBoundaryNodes) <= 2:
			return unorderedBoundaryNodes
		prev = unorderedBoundaryNodes[0]
		cur = self.getBoundaryNeighbors(prev)[0]
		result = [prev, cur]
		while True:
			a, b = self.getBoundaryNeighbors(cur)
			next = a if a != prev else b
			if next == result[0]:
				return result
			result.append(next)
			prev = cur
			cur = next
		#
	def getEdgeColorings(self) :
		return map(lambda coloring: map(lambda node: node.color, coloring), \
				self.generatePossibleEdgeColorings())
	def generatePossibleEdgeColorings(self) :
		"""
		A generator that produces all possible colorings of the boundary nodes,
		modulo renaming of colors.
		>>> config = Configuration([Node("a", 3)], [])
		>>> config.getEdgeColorings()
		[['R', 'B', 'Y'], ['R', 'B', 'G']]
		>>> config = Configuration([Node("a")], [])
		>>> map(lambda colors: ''.join(colors), config.getEdgeColorings())
		['RBYRY', 'RBYRB', 'RBYRG', 'RBYBY', 'RBYBG', 'RBYGY', 'RBYGB', 'RBRYB', 'RBRYG', 'RBRBY', 'RBRBG', 'RBRGY', 'RBRGB', 'RBGYB', 'RBGYG', 'RBGRY', 'RBGRB', 'RBGRG', 'RBGBY', 'RBGBG']
		>>> config = Configuration([Node("a", 4), Node("b", 4), Node("c", 4)],
		...                        [("a", "b"), ("b", "c"), ("a", "c")])
		>>> config.getEdgeColorings()
		[['R', 'B', 'Y'], ['R', 'B', 'G']]
		"""
		cycle = self.getBoundaryCycle()
		assert len(cycle) >= 2, "Boundary too small"
		cycle[0].color = COLORS[0]
		cycle[1].color = COLORS[1]
		index = 2
		while True :
			if index < 2 :
				break
			if index >= len(cycle) :
				yield cycle
				index -= 1
				continue
			colorsToTry = list(set(self.allowedColors(cycle[index])) - \
											set(cycle[index].colorsTried))
			if len(colorsToTry) == 0 :
				cycle[index].color = None
				cycle[index].colorsTried = []
				index -= 1
				continue
			cycle[index].color = colorsToTry[0]
			cycle[index].colorsTried.append(colorsToTry[0])
			index += 1
		cycle[0].color = None
		cycle[1].color = None
		#
	def isColorable(self, retainTheValidColoring = False) :
		"""
		Return True if the is at least one valid coloring of the
		non-boundary nodes.
		"""
		uncoloredNodes = [node for node in self.nodes.values()
						  if node.color == None]
		if len(uncoloredNodes) == 0 :
			return True
		if len(self.allowedColors(uncoloredNodes[0])) == 0 :
			return False
		for color in self.allowedColors(uncoloredNodes[0]) :
			uncoloredNodes[0].color = color
			if self.isColorable(retainTheValidColoring) :
				if not retainTheValidColoring:
					uncoloredNodes[0].color = None
				return True
		uncoloredNodes[0].color = None
		return False
	#
	def clearColoring(self) :
		for node in self.nodes.values() :
			node.color = None
	#
	def isAreducible(self) :
		"""
		Return true if the configuration is A-reducible.
		 In this case A-reducible means that any valid colorings of the
		 Boundarynodes allow a valid coloring for the whole configuration
		 without anything as complex as a Kempe-chain.
		Use:
		>>> config = Configuration([Node("a", 3)], [])
		>>> config.isAreducible()
		True
		>>> config = Configuration([Node("a")], [])
		>>> config.isAreducible()
		False
		>>> config = Configuration([Node("a"), Node("b"), Node("c")],
		...                        [("a", "b"), ("b", "c"), ("c", "a")])
		>>> config.isAreducible()
		False
		>>> config = Configuration([Node("a"), Node("b"), Node("c"), Node("d")],
		...                        [("a", "b"), ("b", "c"), ("c", "d"),
		...                         ("d", "a"), ("b", "d")])
		>>> config.isAreducible()
		False
		"""
		# make a deep-copy of self to make the api externally functional
		testConfig = copy.deepcopy(self)
		for _ in testConfig.generatePossibleEdgeColorings() :
			if not testConfig.isColorable() :
				return False
		return True
	def toDotGraph(self) :
		"""
		Returns a text string in GraphViz .dot format.
		It works pretty well with the 'neato' layout engine.
		>>> config = Configuration([Node("a"), Node("b"), Node("c")],
		...                        [("a", "b"), ("b", "c"), ("c", "a")])
		>>> sys.stdout.write(config.toDotGraph())
		graph {
		node [style="filled"]
		"a_5" []
		"b_5" []
		"c_5" []
		"a_5" -- "b_5"
		"a_5" -- "c_5"
		"b_5" -- "c_5"
		}
		>>> config.isColorable(True)
		True
		>>> sys.stdout.write(config.toDotGraph())
		graph {
		node [style="filled"]
		"a_5" [fillcolor="#ffff00"]
		"b_5" [fillcolor="#0000ff"]
		"c_5" [fillcolor="#ff0000"]
		"a_5" -- "b_5"
		"a_5" -- "c_5"
		"b_5" -- "c_5"
		}
		>>> config.addBoundary()
		>>> config.toDotGraph().count("\\n")
		30
		"""
		results = []
		results.append('graph {\nnode [style="filled"]\n')
		for node in sorted(self.nodes.values()) :
			opts = []
			if node.color != None :
				opts.append('fillcolor="%s"' % colorToCSSColor(node.color))
			if isinstance(node, BoundaryNode) :
				opts.append('style="filled,dashed"')
			results.append('"%s" [%s]\n' % (str(node), ','.join(opts)))
		for nodeA, nodeB in sorted(self.adjacencyList) :
			results.append('"%s" -- "%s"\n' % (str(nodeA), str(nodeB)))
		results.append("}\n")
		return ''.join(results)

if __name__ == "__main__" :
	import doctest
	doctest.testmod()

