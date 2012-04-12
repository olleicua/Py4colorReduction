
import itertools

class GraphFailedPreconditionError(Exception) :
	def __init__(self, value) :
		self.value = value
	def __str__(self) :
		return repr(self.value)

class Node :
	"""
	Use:
	>>> Node("a")
	a_5
	>>> Node("b", 7)
	b_7
	>>> Node("c", 0)
	Traceback (most recent call last):
	    ...
	ValueError: degree must be at least 5
	"""
	def __init__(self, name, degree=5) :
		self.name = name
		if degree < 5 :
			raise ValueError("degree must be at least 5")
		self.degree = degree
		self.color = None

	def __repr__(self) :
		return "%s_%d" % (self.name, self.degree)

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
			self.adjacencyList.append(tuple(sorted((self.nodes[nodeA], self.nodes[nodeB]))))
	
	def __getitem__(self, name) :
		return self.nodes[name]
	
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

	def apparentDegree(self, node) :
		"""
		Use:
		>>> config = Configuration([Node("a"), Node("b", 7), Node("c", 8)],
		...                        [("a", "b"), ("a", "c")])
		>>> config.apparentDegree(config["a"])
		2
		"""
		return len(self.getNeighbors(node))
	
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
			return True
		return node.degree > len(self.getNeighbors(node))
	
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

		outerNodes = self.outerNodes()

		# special-case for len(outerNodes) == 1
		# because a node does not have an edge going to itself,
		# and checking here is faster than checking in consistent().
		if len(outerNodes) <= 1 :
			return outerNodes

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
		
		for ordering in itertools.permutations(outerNodes) :
			if consistent(ordering) :
				return list(ordering)
		raise GraphFailedPreconditionError("no outer node cycle found")
	
	def addBoundary(self) :
		"""
		Add boundary nodes (assumes that the configuartion is triangulated).
		 In cases like:
		     (5) -- (7) -- (5)
		 the behavior of this function will be undefined.
		Use:
		>>> config = Configuration([Node("a")], [])
		>>> config.addBoundary()
		>>> sorted(config.adjacencyList)
		[(a_5, b2|), (a_5, b1|), (a_5, b0|), (a_5, b3|), (a_5, b4|), (b2|, b1|), (b2|, b3|), (b1|, b0|), (b3|, b4|), (b4|, b0|)]
		"""
		# TODO : find a way to get the "outer neighbors of a node"
		outerNodes = self.outerNodes()
		for node in outerNodes :
			newCount = node.degree - self.apparentDegree(node)
			newBoundaryNodes = [BoundaryNode() for n in range(newCount)]
			for i, boundaryNode in enumerate(newBoundaryNodes) :
				self.nodes[boundaryNode.name] = boundaryNode
				self.addEdge(boundaryNode, node)
				if i > 0 :
					self.addEdge(boundaryNode, newBoundaryNodes[i - 1])
			outerNeighbors = [node for n in self.getNeighbors(node) \
							  if self.isOuter(n)]
			#print self.getNeighbors(node)
			self.addEdge(outerNeighbors[0], newBoundaryNodes[0])
			if outerNeighbors[0] != outerNeighbors[1] :
				self.addEdge(outerNeighbors[1], newBoundaryNodes[-1])

if __name__ == "__main__" :
	import doctest
	doctest.testmod()
