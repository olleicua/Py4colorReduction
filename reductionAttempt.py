"""

TYPES, described in prose because Python does not have a way to
describe them all formally.

color: One of the four colors that we are trying to color with.
  Represented as strings "R", "G", "B", "Y".
  Their order (when applicable) is defined by list COLORS and
      function colorSorted.
  (TODO: make it a class with cmp, hash, repr?)

node: A node in the configuration.
  It has a name (string) and conceptual degree
  (which may be higher than the number of edges,
  in the absence of an explicit boundary ring of nodes).
  Represented by an object (class Node).

boundary node: A node in the boundary ring.
  It has a name (string) and cycle number.
  (Every boundary ring is a cycle of nodes,
  so we number them for convenience, picking some
  arbitrary cycle-direction and starting node.
  It's not completely arbitrary; see "addBoundary"
  for details.)
  Represented by an object (class BoundaryNode).

Nodes and boundary nodes are usually interchangable
in the following.  Their sorting is by name.

configuration: A collection of nodes and edges.
  (Nodes may or may not include boundary nodes.)
  Nodes are a dict from name to node.
  Edges are a list of pairs of nodes.  The pairs
  are sorted by node.  The list is unsorted.
  Edges from a node to itself, or multiple edges
  between the same two nodes, are forbidden.
  Every node mentioned by an edge must be in the nodes dict.
  Represented by class Configuration.

coloring: a dict from node to color.
  It does not have to include all nodes in a configuration;
  for example, we have colorings of just boundary nodes
  before we use them to see if we can color the rest
  of the nodes in that case.

color pairs:

kempe chain: a set of boundary nodes that (in a given possibility)
      is connected in the imaginary rest of the graph by alternating
      nodes of the same color-pair (that's one of this possibility's
      color-pairs).
  Its meaning depends on which color-pair it is for.
  Represented by a frozenset of nodes.

kempe chain set: a consistent set of kempe chains that collectively
      include all boundary nodes and are one possibility.
  Its meaning depends on which color-pair it is for.
  Represented by a frozenset of kempe chains.

kempe chain set possibilities: in the context of a configuration,
      a colored boundary ring, and a color-pair, this is a set of
      all the different possible kempe chain sets, at least one of
      which must be true.
  Its meaning depends on which color-pair it is for.
  Represented by a list of kempe chain sets.

"""

import itertools, copy, sys, os

# UTILS #

# from http://stackoverflow.com/questions/2931672/what-is-the-cleanest-way-to-do-a-sort-plus-uniq-on-a-python-list
def sort_uniq(sequence):
	"Return a generator of the given values sorted sans duplicates."
	return (x[0] for x in itertools.groupby(sorted(sequence)))

# from http://docs.python.org/library/itertools.html#recipes
def powerset(iterable):
	"powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
	s = list(iterable)
	return itertools.chain.from_iterable(
			itertools.combinations(s, r) for r in range(len(s)+1))

def valuesSortedByKeys(dict) :
	result = []
	for k in sorted(dict.iterkeys()) :
		result.append(dict[k])
	return result

# CONSTANTS #

COLOR_PAIRS = ["RB", "RG", "RY"]
COLORS = ["R", "G", "B", "Y"]

# colorSorted():
# Or should we just order the colors alphabetically B,G,R,Y?
# Or refer to them by number and have to explicitly translate to
# color-name in the tests and such?

def colorSorted(colorCollection) :
	return [color for color in COLORS if color in colorCollection ]

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
		#
	def __repr__(self) :
		return "%s_%d" % (self.name, self.degree)
	#
	def __cmp__(self, other) :
		"""
		Compares by name (and then id if name is same).
		This makes sorting look nicer and be consistent
		(for algorithms, especially with boundary nodes).
		
		We compare by id after name (lexicographically) so that
		same-name-different-identity nodes (if any)
		compare different so that __hash__ can be correct
		(equal hash when cmp equal, and hash non-mutable).
		"""
		return cmp((self.name, id(self)), (other.name, id(other)))
	#
	def __hash__(self) :
		return id(self)

class BoundaryNode :
	"""
	A node with unspecified degree for the boundary ring of a configuration.
	Use:
	>>> BoundaryNode('b#3')
	b#3
	"""
	def __init__(self, name) :
		self.name = name
		self.cycleNumber = None
		#
	def __repr__(self) :
		return self.name
	#
	def __cmp__(self, other) :
		return cmp((self.name, id(self)), (other.name, id(other)))
	#
	def __hash__(self) :
		return id(self)
	
class Configuration :
	"""
	The initial configurations should be already triangulated as we have no
	 clearly defined algorithm to triangulate them.
	"""
	def __init__(self, nodes = [], adjacencyList = []) :
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
			a = self.nodes[nodeA] if isinstance(nodeA, str) else nodeA
			b = self.nodes[nodeB] if isinstance(nodeB, str) else nodeB
			self.addEdge(a, b)
		#
	def __getitem__(self, name) :
		return self.nodes[name]
	#
	def addEdge(self, nodeA, nodeB) :
		"""
		Use:
		>>> config = Configuration([Node("a"), Node("b", 7)], [])
		>>> config.addEdge(config["a"], config["a"])
		>>> config.adjacencyList
		[]
		>>> config.addEdge(config["a"], config["b"])
		>>> config.adjacencyList
		[(a_5, b_7)]
		>>> config.addEdge(config["a"], config["b"])
		>>> config.adjacencyList
		[(a_5, b_7)]
		"""
		if nodeA != nodeB and not tuple(sorted((nodeA, nodeB))) in self.adjacencyList :
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
	def addNode(self, node) :
		"""
		Adds a node to the graph.
		>>> config = Configuration()
		>>> config.addNode(Node("a"))
		>>> config.addNode(Node("b"))
		>>> sorted(config.nodes.items())
		[('a', a_5), ('b', b_5)]
		>>> config.addNode(Node("a"))
		Traceback (most recent call last):
		    ...
		AssertionError: no duplicate nodes!
		"""
		assert node.name not in self.nodes, "no duplicate nodes!"
		self.nodes[node.name] = node
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
		>>> config.mergeNodes(config.nodes["b"], config.nodes["c"])
		>>> sorted(config.adjacencyList)
		[(b_5, d_5)]
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
	def renameNode(self, oldName, newName) :
		"""
		>>> config = Configuration([Node("a"), Node("b")], [("a", "b")])
		>>> config.renameNode("a", "a")

		>>> config.renameNode("a", "c")
		>>> sorted(config.nodes.keys())
		['b', 'c']
		>>> config.adjacencyList
		[(b_5, c_5)]
		>>> config.renameNode("c", "b")
		Traceback (most recent call last):
		    ...
		AssertionError: renaming c to an existing node name b
		"""
		assert oldName in self.nodes, \
			"renaming a nonexistent node %s to %s" % (oldName, newName)
		if oldName == newName:
			return
		assert newName not in self.nodes, \
			"renaming %s to an existing node name %s" % (oldName, newName)
		node = self.nodes[oldName]
		del self.nodes[oldName]
		node.name = newName
		self.nodes[newName] = node
		# adjacencyList has nodes, not names, so it doesn't have to be changed
		# BUT WAIT its consistency will be broken because of the ordering
		self.adjacencyList = [tuple(sorted(pair)) for pair in self.adjacencyList]
	#
	def isEdge(self, a, b) :
		"""
		Use:
		>>> config = Configuration([Node("a"), Node("b"), Node("c")], [("a", "b")])
		>>> config.isEdge(config["a"], config["b"])
		True
		>>> config.isEdge(config["a"], config["c"])
		False
		>>> config.isEdge(config["a"], config["a"])
		False

		The last example is particularly relevant for coloring:
		it would be a travesty if a node had to be a different
		color as itself.
		"""
		return tuple(sorted((a,b))) in self.adjacencyList
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
		Returns a list of all BoundaryNodes neigboring 'node'.
		The order of the list is as follows:
		
		If node is a boundary node, returns a list
		[prior, next] boundary node in the canonical
		boundary cycle order.
		
		Otherwise, returns a list of all node's boundary-node
		neighbors in some boundary-cycle-consistent order
		(not necessarily the most sensible order;
		this latter constraint only ensures that if
		there are three or more nearby nodes, they won't
		be completely mixed up).

		Use:
		>>> config = Configuration([Node("a"), Node("b", 7)],
		...                        [("a", "b")])
		>>> bn = BoundaryNode('b#x')
		>>> config.nodes[bn.name] = bn
		>>> config.addEdge(config["a"], bn)
		>>> config.getBoundaryNeighbors(config["a"]) == [bn]
		True
		>>> config = Configuration([Node("a", 4)], [])
		>>> config.addBoundary()
		>>> config.getBoundaryNeighbors(config.nodes['b#1'])
		[b#0, b#2]
		>>> config.getBoundaryNeighbors(config.nodes['b#2'])
		[b#1, b#3]
		>>> config.getBoundaryNeighbors(config.nodes['b#3'])
		[b#2, b#0]
		>>> config.getBoundaryNeighbors(config.nodes['b#0'])
		[b#3, b#1]
		>>> config.getBoundaryNeighbors(config.nodes['a'])
		[b#0, b#1, b#2, b#3]
		"""
		result = []
		for nodeA, nodeB in self.adjacencyList :
			if node == nodeA and isinstance(nodeB, BoundaryNode) :
				result.append(nodeB)
			if node == nodeB and isinstance(nodeA, BoundaryNode) :
				result.append(nodeA)
		result.sort()
		try :
			if node.cycleNumber == 0 or \
			   (node.cycleNumber != 1 and result[0].cycleNumber == 0) :
			   	   result.reverse()
		except :
			pass
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
	def allowedColors(self, node, coloring) :
		"""
		>>> config = Configuration([Node("a"), Node("b")],
		...                        [("a", "b")])
		>>> coloring = { config["a"]: "R" }
		>>> sorted(config.allowedColors(config["b"], coloring))
		['B', 'G', 'Y']
		"""
		return set(COLORS) - set(map(lambda node: coloring.get(node),
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
		
		The result will be the first possible result measured by
		lexicographic order by node name.  For example, if "a" is
		the first outer node by name, it will be the first in
		outerNodeCycle, and it will prefer "b" to be the next
		rather than "c", thus determining which way around the
		cycle goes.
		
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
		...                        # A pentagon with centre node "a".
		...                        [Node("a"), Node("b"), Node("q"), Node("d"),
		...                         Node("e"), Node("f")],
		...                        [("a", "b"), ("a", "q"), ("a", "d"),
		...                         ("a", "e"), ("a", "f"),
		...                         ("b", "q"), ("q", "d"), ("d", "e"),
		...                         ("e", "f"), ("f", "b")])
		>>> config.outerNodeCycle()
		[b_5, f_5, e_5, d_5, q_5]
		
		The length of the list is the sum, over each outer node N, of
		the number of connected components the graph of "outer nodes"
		would contain if node N were to be removed.  In other words,
		chokepoints ("cutvertices") have to be traversed multiple times
		in order to get to all parts of the graph.  (My intuition
		claims this, but I have not proved this. --Isaac)

		There exists no correct answer for blobs of nodes whose plane
		representation has at least two separate "outer" regions,
		such as an annulus (ring, tire, etc).
		>>> config = Configuration(
		...              [Node("a",7), Node("b",7), Node("c",7), #outer ring
		...               Node("d",6), Node("e",6), Node("f",6), #non-outer
		...               Node("g",7), Node("h",7), Node("i",7)],#inner outer
		...              [ #each ring
		...               ("a", "b"), ("b", "c"), ("c", "a"),
		...               ("d", "e"), ("e", "f"), ("f", "d"),
		...               ("g", "h"), ("h", "i"), ("i", "g"),
		...                #interconnections outer--non-outer
		...               ("a", "d"), ("a", "e"), ("b", "e"),
		...               ("b", "f"), ("c", "f"), ("c", "d"),
		...                #interconnections non-outer--"inner"
		...               ("g", "d"), ("g", "e"), ("h", "e"),
		...               ("h", "f"), ("i", "f"), ("i", "d")])
		>>> config.outerNodeCycle()
		Traceback (most recent call last):
		    ...
		GraphFailedPreconditionError: 'no outer node cycle found'
		
		
		TODO: The present algorithm is broken for configurations that
		have a cutvertex:

		->>> config = Configuration(
		...                        # Two triangles that share a node "a".
		...                        [Node("a",6), Node("b"), Node("c"),
		...                         Node("d"), Node("e")],
		...                        [("a", "b"), ("a", "c"), ("b", "c"),
		...                         ("a", "d"), ("a", "e"), ("d", "e")])
		->>> config.outerNodeCycle()
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
		Add boundary nodes (assumes that the configuration is
		triangulated).

		The boundary nodes start next to the lexicographically first
		outer node and point in the direction towards the next
		(assuming there are at least three outer nodes; otherwise
		this is meaningless).  The first node in the boundary cycle
		is, given that direction, the earliest one that's joined to
		the lexicographically first outer node.

		In cases like:
		    (5) -- (7) -- (5)
		the behavior of this function will be undefined.

		Use:
		>>> config = Configuration([Node("a")], [])
		>>> config.addBoundary()
		>>> len(config.getBoundaryNeighbors(config["a"])) == 5
		True
		>>> sorted(config.adjacencyList)
		[(a_5, b#0), (a_5, b#1), (a_5, b#2), (a_5, b#3), (a_5, b#4), (b#0, b#1), (b#0, b#4), (b#1, b#2), (b#2, b#3), (b#3, b#4)]
		
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
		>>> config = Configuration([Node("a", 3), Node("b"), Node("c")],
		...                        [("a", "b"), ("b", "c"), ("a", "c")])
		>>> config.addBoundary()
		>>> len(config.nodes)
		7
		>>> len(config.adjacencyList)
		14
		"""
		if any([isinstance(node, BoundaryNode) for node in self.nodes.values()]) :
			return
		tentativeBoundaryNumber = [0]
		def makeBoundaryNode() :
			name = "b#tentative" + str(tentativeBoundaryNumber[0])
			tentativeBoundaryNumber[0] += 1
			node = BoundaryNode(name)
			self.nodes[name] = node
			return node
		# add triangulating boundary nodes between each pair in the outer cycle
		outerCycle = self.outerNodeCycle()
		for cycleI, node in enumerate(outerCycle) :
			newBoundaryNode = makeBoundaryNode()
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
			elif 0 == newCount :
				# degree satisfied; join triangleNodes
				# (if they're distinct nodes)
				if 2 == len(triangleNodes) :
					self.addEdge(*triangleNodes)
				#
			else :
				# generate new BoundaryNodes
				newBoundaryNodes = [makeBoundaryNode() for _ in range(newCount)]
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
					self.addEdge(boundaryNode, node)
					if newBNi > 0 :
						self.addEdge(boundaryNode, newBoundaryNodes[newBNi - 1])
		unorderedBoundaryNodes = sorted([node for node in self.nodes.values() if
							isinstance(node, BoundaryNode)])
		if len(unorderedBoundaryNodes) <= 2:
			orderedBoundaryNodes = unorderedBoundaryNodes
		else:
			prev = unorderedBoundaryNodes[0]
			cur = self.getBoundaryNeighbors(prev)[0]
			orderedBoundaryNodes = [prev, cur]
			while True:
				a, b = self.getBoundaryNeighbors(cur)
				next = a if a != prev else b
				if next == orderedBoundaryNodes[0]:
					break
				orderedBoundaryNodes.append(next)
				prev = cur
				cur = next
		for i, node in enumerate(orderedBoundaryNodes) :
			node.cycleNumber = i
			self.renameNode(node.name, "b#" + str(i))
		#
	def getBoundaryCycle(self) :
		"""
		Returns a list of the boundary nodes in cyclic order.
		Automatically adds a boundary if it isn't there already.
		
		If there are extra BoundaryNodes not in the cycle,
		returns them too, sorted in lexicographic order by name
		(generatePossibleBoundaryColorings* currently rely on this).
		
		>>> config = Configuration([Node("a"), Node("b"), Node("c"), Node("d")],
		...                        [("a", "b"), ("b", "c"), ("c", "d"),
		...                         ("d", "a"), ("b", "d")])
		>>> len(config.getBoundaryCycle())
		6
		>>> len([pair for pair in config.adjacencyList
		...        if isinstance(pair[0], BoundaryNode) and isinstance(pair[1], BoundaryNode)])
		6
		>>> config = Configuration([Node("a", 4)], [])
		>>> len(config.getBoundaryCycle())
		4
		>>> config.getBoundaryNeighbors(config.nodes["b#3"])
		[b#2, b#0]
		>>> config = Configuration([Node("a", 3)], [])
		>>> len(config.getBoundaryCycle())
		3
		>>> config = Configuration([Node("a", 2)], [])
		>>> len(config.getBoundaryCycle())
		2
		>>> config = Configuration([Node("a", 1)], [])
		>>> len(config.getBoundaryCycle())
		1
		>>> config = Configuration([Node("a", 0)], [])
		>>> len(config.getBoundaryCycle())
		0
		"""
		self.addBoundary()
		return sorted([node for node in self.nodes.values() if
					isinstance(node, BoundaryNode)])
		#
	def generatePossibleBoundaryColorings(self) :
		"""
		A generator that produces all possible colorings of the boundary nodes,
		modulo renaming of colors.
		>>> config = Configuration([Node("a", 3)], [])
		>>> def getColorings(config) :
		...     return [valuesSortedByKeys(coloring) for coloring in config.generatePossibleBoundaryColorings()]
		>>> getColorings(config)
		[['R', 'G', 'B']]
		>>> config = Configuration([Node("a")], [])
		>>> sorted(map(lambda colors: ''.join(colors), getColorings(config)))
		['RGBGB', 'RGBGY', 'RGBRB', 'RGBRG', 'RGBRY', 'RGBYB', 'RGBYG', 'RGRBG', 'RGRBY', 'RGRGB']
		>>> config = Configuration([Node("a", 4), Node("b", 4), Node("c", 4)],
		...                        [("a", "b"), ("b", "c"), ("a", "c")])
		>>> getColorings(config)
		[['R', 'G', 'B']]

		"""
		cycle = self.getBoundaryCycle()
		return self.findColorings(skipSameUpToColorRenaming = True, nodesToColor = cycle)
	def generatePossibleBoundaryColoringsGivenSmallerConfiguration(self,
			setsOfBoundaryNodesToMerge = [], newEdges = [], newNodes = []) :
		cycle = self.getBoundaryCycle()
		testConfig = Configuration(
				cycle,
				filter(lambda (a,b): isinstance(a,BoundaryNode)
				                 and isinstance(b,BoundaryNode),
				       self.adjacencyList))
		mapFromMainToTestNodes = {}
		for node in cycle :
			mapFromMainToTestNodes[node] = node
		# The order of adding nodes, adding edges, then
		# deleting(merging) nodes is important here.
		for node in newNodes :
			testConfig.addNode(node)
		for a,b in newEdges :
			testConfig.addEdge(a, b)
		for nodeSet in setsOfBoundaryNodesToMerge :
			nodes = list(set(nodeSet))
			base = nodes[0]
			rest = nodes[1:]
			for node in rest :
				assert not self.isEdge(base, node), """
					Nodes that must be different colors due to an edge
					cannot be merged to have the same color as each other!
					"""
				testConfig.mergeNodes(base, node)
				mapFromMainToTestNodes[node] = base
		assert len(testConfig.nodes) < len(self.nodes), \
			"This config isn't smaller, so the math proof usefulness of this boundary-coloring is nil."
		# TODO can we also test the planarity of this test config, and
		# the original-nodes-are-still-all-connected-to-the-infinite-region
		# -ness of it?
		# What about the math problems with large rings, node merging, and
		# whether they were the same nodes already?
		for testColoring in testConfig.generatePossibleBoundaryColorings() :
			coloring = {}
			for main, test in mapFromMainToTestNodes.iteritems() :
				coloring[main] = testColoring[test]
			yield coloring
	#
	def generatePossibleBoundaryColoringsTryingCReduction(self) :
		cycle = self.getBoundaryCycle()
		newNodes = []
		newEdges = []
		setsOfBoundaryNodesToMerge = []
		if len(self.nodes) - len(cycle) > 1 :
			centerNode = BoundaryNode("b#center") #only valid for config size > 1
			newNodes.append(centerNode)
			for node in cycle :
				newEdges.append((centerNode, node))
		elif len(cycle) > 0 :
			for node in cycle :
				newEdges.append((cycle[0], node))
		if len(cycle) >= 4 :
			setsOfBoundaryNodesToMerge.append([cycle[0], cycle[2]])
		#
		return self.generatePossibleBoundaryColoringsGivenSmallerConfiguration(
				setsOfBoundaryNodesToMerge, newEdges, newNodes)
	#
	def findKempeChainSetPossibilities(self, boundaryColoring, colorPair) :
		"""
		Automatically adds a boundary if it isn't there already.
		Results in a large number of sets like, if there are five
		boundary nodes a-e, perhaps
		->>> config = Configuration([Node("a"), Node("b"), Node("c"), Node("d")],
		-...                        [("a", "b"), ("b", "c"), ("c", "d"),
		-...                         ("d", "a"), ("b", "d")])
		>>> config = Configuration([Node("a", 4)], [])
		>>> for coloring in config.generatePossibleBoundaryColorings():
		...     print valuesSortedByKeys(coloring), ":"
		...     def toList(setss): return sorted(sorted(sorted(set) for set in sets) for sets in setss)
		...     r,g,b,y = COLORS
		...     print r+g+'/'+b+y+':', \
		            toList(config.findKempeChainSetPossibilities(coloring, (r, g)))
		...     print r+b+'/'+g+y+':', \
		            toList(config.findKempeChainSetPossibilities(coloring, (r, b)))
		...     print r+y+'/'+g+b+':', \
		            toList(config.findKempeChainSetPossibilities(coloring, (r, y)))
		..yeah.

		[set([a, c]), set([b]), set([d, e])]
		{a: 0, b: 1, c: 0, d: 2, e: 2}
		{a: set([a, c]), b: set([b]), c: set([a, c]), d: set([d, e]), e: set([d, e])}
		: a list of one of those.
		
		each of which (for the given pair of pair of colors)
		is in this possibility Kempe-chain-connected and thus
		can have its two colors exchanged anytime.
		...i'll just make it return a list and check whether it gets too large. simpler coding.
		"""
		colorPair1 = set(colorPair)
		colorPair2 = set(COLORS) - colorPair1
		def whichColorPair(node) :
			"""Returns a value comparable for equality which determines
			   an equivalence class between colors."""
			return boundaryColoring[node] in colorPair1
		assert len(colorPair1) == len(colorPair2) == 2, \
				"Kempe chains work with pairs of colors"
		cycle = self.getBoundaryCycle()
		result = []
		def weFoundAResult(kempeChains) :
			# temp dict from chain number to set of nodes in the chain
			chains = {}
			for cycleIndex, chainNumber in kempeChains.iteritems() :
				chains.setdefault(chainNumber, set()).add(cycle[cycleIndex])
			kempeChainSet = frozenset(frozenset(chain) for chain in chains.values())
			result.append(kempeChainSet)
		#
		# nested function
		def tryNext(i, kempeChainsSoFar, numKempeChainsSoFar, occludedSet) :
			"""
			Recursively look for kempe grouping possibilities for each of the
			 remaining nodes in the cycle.
			i:
				This is the 'cycle'-index of the current node we're
				looking at.  Every recursive call increments i.
				The recursion ends when i == len(cycle).
			kempeChainsSoFar:
				This is a dictionary from 'cycle'-index (integer) to
				kempe-chain-index (integer) that contains keys range(0,i)
				(plus possibly range(n,len(cycle)) for some n due to
				"horrible corner case"), and its values contains each of
				range(0,numKempeChainsSoFar) at least once.
			numKempeChainsSoFar:
				This integer just counts the number of Kempe chains we've named.
			occludedSet: Set of 'cycle'-indices of nodes we've seen for which
				there is a kempe chain we've seen that's attached to
				at least one node before and one node after the occluded node.
				Any new node we look at, since we look at them in order,
				will not be able to attach to it in a kempe chain (except
				possibly by attaching to one of those before or after nodes,
				which we allow when possible, i.e., when they themselves are
				not occluded by yet another chain).
			"""
			if i == len(cycle) :
				weFoundAResult(kempeChainsSoFar)
				return
			#
			# sub-nested function
			def joinKempeChainAndRecur(kempeChainID) :
				newKempeChainsSoFar = copy.copy(kempeChainsSoFar)
				newOccludedSet = copy.copy(occludedSet)
				newKempeChainsSoFar[i] = kempeChainID
				newKempeChain = [n for n in newKempeChainsSoFar.keys() \
				                         if newKempeChainsSoFar[n] == kempeChainID ]
				newlyOccluded = range(min(newKempeChain)+1, i)
				newOccludedSet |= set(newlyOccluded)
				tryNext(i + 1, newKempeChainsSoFar, numKempeChainsSoFar, newOccludedSet)
			#
			# end sub-nested function: tryNext
			#
			if i in kempeChainsSoFar :
				# finish dealing with the horrible corner case mentioned below
				weFoundAResult(kempeChainsSoFar)
			elif whichColorPair(cycle[i - 1]) == whichColorPair(cycle[i]) :
				prevNodeIndex = (i - 1) % len(cycle)
				prevNodeKempeChain = kempeChainsSoFar[prevNodeIndex]
				joinKempeChainAndRecur(prevNodeKempeChain)
			else :
				priorNodesThatWeCanConnectTo = \
					       [n for n in range(0, i) \
						 if n not in occludedSet and \
						  whichColorPair(cycle[n]) == \
						  whichColorPair(cycle[i])]
				kempeChainIDsThatWeCanConnectTo = list(set([ \
					kempeChainsSoFar[n] for n in priorNodesThatWeCanConnectTo]))
				#
				for kempeChainID in kempeChainIDsThatWeCanConnectTo:
					joinKempeChainAndRecur(kempeChainID)
				newSingletonKempeChainID = numKempeChainsSoFar
				numKempeChainsSoFar += 1
				joinKempeChainAndRecur(newSingletonKempeChainID)
		#
		# end nested function: joinInKempeChainAndRecur
		#
		startingKempeChains = {0: 0}
		# Deal with a horrible corner case:
		for i in reversed(xrange(len(cycle))) :
			if whichColorPair(cycle[i]) == whichColorPair(cycle[0]) :
				startingKempeChains[i] = 0
			else:
				break
		tryNext(1, startingKempeChains, 1, set())
		return result
	#
	def isColorable(self, givenColoringOfSomeNodes = {}) :
		"""
		Return True if the is at least one valid coloring of the
		non-boundary nodes.
		"""
		for coloring in self.findColorings(givenColoringOfSomeNodes, True) :
			return True
		return False
	def findColoring(self, givenColoringOfSomeNodes = {}) :
		"""
		Return a valid coloring, or None if none exists.
		"""
		for coloring in self.findColorings(givenColoringOfSomeNodes, True) :
			return coloring
		return None
	def findColorings(self, givenColoring = {}, skipSameUpToColorRenaming = False, nodesToColor = None) :
		if nodesToColor == None :
			nodesToColor = [node for node in valuesSortedByKeys(self.nodes) if node not in givenColoring]
		if skipSameUpToColorRenaming :
			# verbotenColors is a list so it's ordered by
			# canonical color order (which is nicer). It's
			# short enough that speed difference is unimportant.
			# It contains all currently-used colors plus one
			# not-yet-used color (which is equivalent to the
			# other unused colors, thus those other unused colors
			# are arbitrarily verboten until the first unused
			# color is used so arbitrarily the next one is
			# then not verboten).
			usedColors = set(givenColoring.itervalues())
			verbotenColors = [color for color in COLORS if color not in usedColors][1:]
		else:
			usedColors = None
			verbotenColors = None
		return self.findColoringsImpl(givenColoring, nodesToColor, skipSameUpToColorRenaming,
				usedColors, verbotenColors)
	def findColoringsImpl(self, givenColoring, nodesToColor, skipSameUpToColorRenaming, usedColors, verbotenColors) :
		if len(nodesToColor) == 0 :
			yield givenColoring
			return
		nextNode = nodesToColor[0]
		nodesToColor = nodesToColor[1:]
		for color in colorSorted(self.allowedColors(nextNode, givenColoring)) :
			newUsedColors = usedColors
			newVerbotenColors = verbotenColors
			if skipSameUpToColorRenaming :
				if color in verbotenColors :
					continue
				elif color not in usedColors :
					newUsedColors = usedColors | set(color)
					newVerbotenColors = verbotenColors[1:]
			tryColoring = givenColoring.copy()
			tryColoring[nextNode] = color
			for coloring in self.findColoringsImpl(tryColoring, nodesToColor, skipSameUpToColorRenaming,
						newUsedColors, newVerbotenColors) :
				yield coloring
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
		for coloring in testConfig.generatePossibleBoundaryColorings() :
			if not testConfig.isColorable(coloring) :
				return False
		return True
	#
	def swapKempeChain(self, coloring, colorPair, kempeChain) :
		result = coloring.copy()
		for boundaryNode in kempeChain :
			color = coloring[boundaryNode]
			if color in colorPair :
				if color == colorPair[0] :
					newColor = colorPair[1]
				else :
					newColor = colorPair[0]
			else : # other pair
				otherPair = list(set(COLORS) - set(colorPair))
				if color == otherPair[0] :
					newColor = otherPair[1]
				else :
					newColor = otherPair[0]
			result[boundaryNode] = newColor
		return result
	#
	def swapKempeChains(self, coloring, colorPair, kempeChains) :
		newColoring = coloring
		for kempeChain in kempeChains:
			newColoring = self.swapKempeChain(newColoring, colorPair, kempeChain)
		return newColoring
	#
	def kempeChainSetAllowsReduction(self, coloring, colorPair, kempeChainSet) :
		for kempeChainsToSwap in powerset(kempeChainSet) :
			newColoring = self.swapKempeChains(coloring, colorPair, kempeChainsToSwap)
			if self.isColorable(newColoring) :
				return True
		return False
	#
	def kempePossibilitiesAllowReduction(self, coloring, colorPair, kempeChainSetPossibilities) :
		for kempeChains in kempeChainSetPossibilities :
			if not self.kempeChainSetAllowsReduction(coloring, colorPair, kempeChains) :
				return False
		return True
	#
	def isDreducible(self, makeGraphViz=False) :
		"""
		Return true if the configuration is D-reducible.
		 In this case D-reducible means that any valid colorings of the
		 BoundaryNodes allow a valid coloring for the whole configuration
		 with simple Kempe-chain arguments.
		 D.A. Holton/J.Sheehan say that Birkhoff diamond is D-recucible, but
		 our code shows counterexamples.
		>>> config = Configuration([Node("a", 4)], [])
		>>> config.isDreducible()
		single node of degree 4
		>>> config = Configuration([Node("a", 5)], [])
		>>> config.isDreducible()
		single node of degree 5
		>>> config = Configuration([Node("a"), Node("b"), Node("c")],
		...                        [("a", "b"), ("b", "c"), ("c", "a")])
		>>> config.isDreducible()
		triangle
		>>> config = Configuration([Node("a"), Node("b"), Node("c"), Node("d")],
		...                        [("a", "b"), ("b", "c"), ("c", "d"),
		...                         ("d", "a"), ("b", "d")])
		>>> config.isDreducible()
		birkhoff diamond
		>>> config = Configuration([Node("a"), Node("b", 6), Node("c"), Node("d", 6)],
		...                        [("a", "b"), ("b", "c"), ("c", "d"),
		...                         ("d", "a"), ("b", "d")])
		>>> config.isDreducible()
		bernhart diamond
		>>> config = Configuration([Node("a"), Node("b"), Node("c"), Node("d", 6)],
		...                        [("a", "b"), ("b", "c"), ("c", "d"),
		...                         ("d", "a"), ("b", "d")])
		>>> config.isDreducible()
		example 2 from rsst
		>>> config = Configuration([Node("a", 6), Node("b", 6), Node("c"), Node("d"), Node("e")],
		...                        [("a", "b"), ("b", "c"), ("c", "d"),
		...                         ("d", "e"), ("e", "a"), ("a", "d"), ("b", "d")])
		>>> config.isDreducible()
		example 5 from rsst
		"""
		testConfig = copy.deepcopy(self)
		for coloring in testConfig.generatePossibleBoundaryColorings() :
			if not testConfig.isColorable(coloring) :
				kempeArgumentFound = False
				for colorPair in COLOR_PAIRS :
					kempeChainSetPossibilities = testConfig.findKempeChainSetPossibilities(coloring, colorPair)
					if testConfig.kempePossibilitiesAllowReduction(coloring, colorPair, kempeChainSetPossibilities) :
						kempeArgumentFound = True
						break
				if not kempeArgumentFound :
					if makeGraphViz :
						open("test.dot", "w+").write(testConfig.toDotGraph())
						os.system("neato -Tpng test.dot > test.png")
					return False
		return True
	#
	def isCreducible(self, makeGraphViz=False) :
		"""
		Return true if the configuration is C-reducible.
		 In this case C-reducible means that any valid colorings of the
		 Boundarynodes allow a valid coloring for the whole configuration
		 with simple Kempe-chain arguments and a simple shrinking of the
		 smaller graph. Specifically, if there are 4 or more boundary
		 nodes, assume the 1st and 3rd boundary nodes are the same node
		 and thus the same color.
		>>> config = Configuration([Node("a", 4)], [])
		>>> config.isCreducible()
		single node of degree 4
		>>> config = Configuration([Node("a", 5)], [])
		>>> config.isDreducible()
		single node of degree 5
		>>> config = Configuration([Node("a"), Node("b"), Node("c")],
		...                        [("a", "b"), ("b", "c"), ("c", "a")])
		>>> config.isCreducible()
		(C) triangle
		>>> config = Configuration([Node("a"), Node("b"), Node("c"), Node("d")],
		...                        [("a", "b"), ("b", "c"), ("c", "d"),
		...                         ("d", "a"), ("b", "d")])
		>>> config.isCreducible()
		(C) birkhoff diamond
		>>> config = Configuration([Node("a"), Node("b", 6), Node("c"), Node("d", 6)],
		...                        [("a", "b"), ("b", "c"), ("c", "d"),
		...                         ("d", "a"), ("b", "d")])
		>>> config.isCreducible(True)
		(C) bernhart diamond
		>>> config = Configuration([Node("a"), Node("b"), Node("c"), Node("d", 6)],
		...                        [("a", "b"), ("b", "c"), ("c", "d"),
		...                         ("d", "a"), ("b", "d")])
		>>> config.isCreducible()
		(C) example 2 from rsst
		>>> config = Configuration([Node("a", 6), Node("b", 6), Node("c"), Node("d"), Node("e")],
		...                        [("a", "b"), ("b", "c"), ("c", "d"),
		...                         ("d", "e"), ("e", "a"), ("a", "d"), ("b", "d")])
		>>> config.isCreducible()
		(C) example 5 from rsst
		"""
		testConfig = copy.deepcopy(self)
		for coloring in testConfig.generatePossibleBoundaryColoringsTryingCReduction() :
			if not testConfig.isColorable(coloring) :
				kempeArgumentFound = False
				for colorPair in COLOR_PAIRS :
					kempeChainSetPossibilities = testConfig.findKempeChainSetPossibilities(coloring, colorPair)
					if testConfig.kempePossibilitiesAllowReduction(coloring, colorPair, kempeChainSetPossibilities) :
						kempeArgumentFound = True
						break
				if not kempeArgumentFound :
					if makeGraphViz :
						open("test.dot", "w+").write(testConfig.toDotGraph(coloring))
						os.system("neato -Tpng test.dot > test.png")
					return False
		return True
	def toDotGraph(self, coloring = {}) :
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
		>>> coloring = config.findColoring()
		>>> coloring != None
		True
		>>> sys.stdout.write(config.toDotGraph(coloring))
		graph {
		node [style="filled"]
		"a_5" [fillcolor="#ff0000"]
		"b_5" [fillcolor="#00ff00"]
		"c_5" [fillcolor="#0000ff"]
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
			if node in coloring :
				opts.append('fillcolor="%s"' % colorToCSSColor(coloring[node]))
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

