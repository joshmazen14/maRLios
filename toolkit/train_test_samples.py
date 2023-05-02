TRAIN_SET = [
	 [['left'], ['A', 'B', 'left']] ,
	 [['B', 'right'], ['NOOP']] ,
	 [['A', 'down'], ['right']] ,
	 [['A', 'B', 'left'], ['down']] ,
	 [['A', 'down'], ['B', 'left']] ,
	 [['A', 'left'], ['A', 'down', 'left']] ,
	 [['A', 'B', 'down', 'right'], ['B', 'down', 'left']] ,
	 [['NOOP'], ['A', 'right']] ,
	 [['A', 'left'], ['NOOP']] ,
	 [['B', 'right'], ['A', 'B', 'right']] ,
	 [['B', 'down'], ['down', 'left']] ,
	 [['A', 'B'], ['B', 'right']] ,
	 [['A'], ['B']] ,
	 [['A', 'B', 'down', 'right'], ['B']] ,
	 [['A'], ['A', 'B', 'down']] ,
	 [['B', 'down', 'right'], ['down', 'right']] ,
	 [['right'], ['B', 'right']] ,
	 [['A', 'left'], ['down', 'right']] ,
	 [['down'], ['B']] ,
	 [['A', 'B'], ['A', 'B', 'down']] ,
	 [['A', 'down', 'right'], ['B', 'down', 'left']] ,
	 [['down'], ['A', 'B', 'left']] ,
	 [['A'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'B', 'left'], ['A', 'down', 'right']] ,
	 [['A', 'B', 'down'], ['B', 'down']] ,
	 [['A', 'B', 'left'], ['B', 'down', 'right']] ,
	 [['A', 'B', 'down', 'left'], ['down', 'left']] ,
	 [['B', 'right'], ['A', 'down', 'right']] ,
	 [['B', 'left'], ['B', 'down', 'right']] ,
	 [['left'], ['A', 'B', 'down', 'left']] ,
	 [['left'], ['B', 'left']] ,
	 [['A', 'down'], ['A', 'left']] ,
	 [['down'], ['A', 'down', 'right']] ,
	 [['A'], ['A', 'left']] ,
	 [['B', 'left'], ['down', 'left']] ,
	 [['A', 'B', 'down'], ['down', 'left']] ,
	 [['A', 'B'], ['A', 'B', 'left']] ,
	 [['left'], ['A', 'left']] ,
	 [['down'], ['right']] ,
	 [['A', 'down', 'left'], ['A', 'B', 'left']] ,
	 [['A'], ['left']] ,
	 [['A', 'left'], ['A', 'B', 'down', 'right']] ,
	 [['B', 'down'], ['A', 'B']] ,
	 [['A', 'left'], ['A', 'right']] ,
	 [['left'], ['A']] ,
	 [['A', 'B', 'down'], ['B', 'down', 'left']] ,
	 [['NOOP'], ['A', 'down']] ,
	 [['A', 'B', 'down', 'right'], ['A', 'down', 'left']] ,
	 [['A', 'B', 'right'], ['left']] ,
	 [['A', 'down', 'left'], ['A']] ,
	 [['A', 'B'], ['A']] ,
	 [['A', 'left'], ['B', 'right']] ,
	 [['A', 'down', 'right'], ['B', 'right']] ,
	 [['right'], ['B', 'left']] ,
	 [['A', 'B', 'down', 'left'], ['down', 'right']] ,
	 [['right'], ['A', 'right']] ,
	 [['B', 'left'], ['A']] ,
	 [['A', 'B', 'down'], ['A', 'right']] ,
	 [['A', 'down', 'right'], ['A', 'B', 'down']] ,
	 [['A', 'left'], ['B', 'left']] ,
	 [['A', 'B', 'down'], ['B']] ,
	 [['right'], ['B', 'down', 'right']] ,
	 [['down', 'left'], ['A', 'B', 'down']] ,
	 [['B', 'down', 'right'], ['A', 'B', 'right']] ,
	 [['down'], ['A']] ,
	 [['down'], ['A', 'B', 'down']] ,
	 [['A', 'left'], ['A', 'down', 'right']] ,
	 [['down', 'left'], ['right']] ,
	 [['A', 'B', 'down', 'right'], ['NOOP']] ,
	 [['A', 'down', 'right'], ['A']] ,
	 [['right'], ['B']] ,
	 [['A', 'B', 'left'], ['A', 'B', 'down']] ,
	 [['NOOP'], ['B', 'down']] ,
	 [['A', 'down', 'left'], ['down', 'left']] ,
	 [['B', 'down', 'left'], ['A', 'right']] ,
	 [['A', 'B', 'left'], ['A', 'B', 'down', 'left']] ,
	 [['down', 'left'], ['B', 'right']] ,
	 [['B', 'down', 'right'], ['A', 'down']] ,
	 [['B', 'down', 'left'], ['B', 'left']] ,
	 [['down', 'right'], ['B', 'down', 'left']] ,
	 [['B'], ['A', 'down', 'right']] ,
	 [['A', 'B', 'down', 'right'], ['A']] ,
	 [['A', 'B', 'down', 'left'], ['NOOP']] ,
	 [['down'], ['B', 'down', 'right']] ,
	 [['right'], ['A', 'B', 'right']] ,
	 [['A', 'B'], ['B']] ,
	 [['A', 'B'], ['A', 'left']] ,
	 [['down', 'left'], ['A', 'right']] ,
	 [['A'], ['A', 'B']] ,
	 [['B', 'down', 'right'], ['A', 'right']] ,
	 [['B', 'down', 'left'], ['A', 'left']] ,
	 [['down', 'right'], ['A', 'down', 'right']] ,
	 [['B', 'down'], ['A', 'down']] ,
	 [['B', 'right'], ['B', 'down', 'left']] ,
	 [['A', 'B'], ['A', 'down', 'left']] ,
	 [['A', 'right'], ['A', 'B']] ,
	 [['down', 'left'], ['A', 'down', 'right']] ,
	 [['B'], ['A', 'B', 'down']] ,
	 [['left'], ['left']] ,
	 [['A', 'B', 'down', 'right'], ['B', 'right']] ,
	 [['B', 'down', 'left'], ['A', 'down', 'left']] ,
	 [['B', 'left'], ['right']] ,
	 [['down', 'right'], ['B']] ,
	 [['B'], ['left']] ,
	 [['A', 'down', 'right'], ['B']] ,
	 [['B'], ['A', 'right']] ,
	 [['right'], ['right']] ,
	 [['NOOP'], ['left']] ,
	 [['A', 'B', 'right'], ['B']] ,
	 [['B', 'down'], ['A', 'right']] ,
	 [['right'], ['down', 'left']] ,
	 [['NOOP'], ['B', 'left']] ,
	 [['A', 'left'], ['A', 'B', 'left']] ,
	 [['down', 'right'], ['A', 'B']] ,
	 [['B', 'left'], ['B', 'right']] ,
	 [['A', 'B', 'down'], ['A', 'down']] ,
	 [['B', 'right'], ['B', 'down']] ,
	 [['A', 'right'], ['A', 'down']] ,
	 [['B', 'right'], ['A', 'left']] ,
	 [['A', 'right'], ['B', 'down', 'right']] ,
	 [['left'], ['down']] ,
	 [['B', 'left'], ['A', 'down', 'left']] ,
	 [['A', 'right'], ['NOOP']] ,
	 [['A', 'B', 'left'], ['A']] ,
	 [['right'], ['A', 'left']] ,
	 [['B', 'down', 'right'], ['right']] ,
	 [['down'], ['A', 'left']] ,
	 [['B'], ['B', 'left']] ,
	 [['A', 'right'], ['A', 'left']] ,
	 [['B', 'down', 'left'], ['down']] ,
	 [['right'], ['B', 'down']] ,
	 [['A', 'B'], ['down', 'left']] ,
	 [['A', 'down', 'right'], ['left']] ,
	 [['down', 'left'], ['A', 'down']] ,
	 [['A', 'B', 'right'], ['B', 'right']] ,
	 [['A', 'down', 'left'], ['right']] ,
	 [['A', 'B', 'down', 'right'], ['B', 'down']] ,
	 [['B', 'down', 'left'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'right'], ['down']] ,
	 [['NOOP'], ['down', 'right']] ,
	 [['A', 'B', 'down'], ['NOOP']] ,
	 [['B', 'down', 'right'], ['A', 'B', 'down']] ,
	 [['A', 'down', 'left'], ['down']] ,
	 [['A', 'B', 'right'], ['B', 'down']] ,
	 [['A', 'B', 'left'], ['A', 'B', 'down', 'right']] ,
	 [['down', 'left'], ['B', 'down', 'left']] ,
	 [['A', 'B', 'left'], ['B', 'right']] ,
	 [['down', 'left'], ['down', 'right']] ,
	 [['A', 'down'], ['down']] ,
	 [['left'], ['down', 'left']] ,
	 [['B', 'down'], ['B', 'down']] ,
	 [['A', 'B', 'down', 'right'], ['B', 'left']] ,
	 [['down', 'right'], ['left']] ,
	 [['A', 'down', 'left'], ['NOOP']] ,
	 [['B', 'down', 'left'], ['B']] ,
	 [['A', 'B'], ['A', 'B', 'down', 'left']] ,
	 [['B', 'right'], ['B']] ,
	 [['A', 'down', 'left'], ['A', 'left']] ,
	 [['A'], ['A', 'B', 'left']] ,
	 [['A', 'B', 'down', 'left'], ['B', 'right']] ,
	 [['B'], ['A', 'left']] ,
	 [['NOOP'], ['down']] ,
	 [['right'], ['A', 'down']] ,
	 [['A', 'down', 'right'], ['A', 'B', 'down', 'right']] ,
	 [['A', 'B', 'down'], ['A', 'B']] ,
	 [['B', 'left'], ['B', 'down', 'left']] ,
	 [['right'], ['NOOP']] ,
	 [['A', 'right'], ['A', 'down', 'left']] ,
	 [['A', 'B', 'down', 'left'], ['B']] ,
	 [['A', 'down'], ['B', 'right']] ,
	 [['A', 'B', 'down'], ['right']] ,
	 [['A', 'B', 'down', 'left'], ['A']] ,
	 [['A', 'left'], ['B']] ,
	 [['A', 'B', 'down', 'right'], ['down', 'left']] ,
	 [['B'], ['B']] ,
	 [['down'], ['A', 'B', 'right']] ,
	 [['A', 'B', 'right'], ['A', 'right']] ,
	 [['down'], ['down', 'right']] ,
	 [['B', 'down', 'right'], ['B', 'down']] ,
	 [['right'], ['A', 'B', 'down', 'right']] ,
	 [['left'], ['A', 'B', 'right']] ,
	 [['B'], ['A', 'B', 'down', 'right']] ,
	 [['left'], ['B']] ,
	 [['A', 'B', 'down'], ['down']] ,
	 [['right'], ['A', 'B']] ,
	 [['B', 'down', 'left'], ['B', 'down', 'left']] ,
	 [['right'], ['A', 'down', 'left']] ,
	 [['A', 'B', 'right'], ['A', 'B', 'left']] ,
	 [['A', 'down'], ['down', 'left']] ,
	 [['A'], ['B', 'down']] ,
	 [['down', 'left'], ['left']] ,
	 [['down', 'right'], ['B', 'down', 'right']] ,
	 [['NOOP'], ['right']] ,
	 [['A'], ['down', 'left']] ,
	 [['A', 'B', 'left'], ['A', 'B', 'left']] ,
	 [['B', 'left'], ['NOOP']] ,
	 [['B'], ['A', 'down']] ,
	 [['A', 'B', 'down', 'left'], ['left']] ,
	 [['down', 'right'], ['A', 'down']] ,
	 [['A', 'left'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'down'], ['A', 'down', 'right']] ,
	 [['A', 'B', 'right'], ['A', 'down']] ,
	 [['A', 'down', 'left'], ['down', 'right']] ,
	 [['A', 'B', 'right'], ['A']] ,
	 [['B', 'left'], ['down']] ,
	 [['A', 'right'], ['A', 'B', 'left']] ,
	 [['right'], ['A', 'down', 'right']] ,
	 [['B', 'left'], ['B', 'down']] ,
	 [['A', 'B'], ['down', 'right']] ,
	 [['B'], ['right']] ,
	 [['NOOP'], ['A', 'B', 'left']] ,
	 [['B', 'down'], ['A', 'left']] ,
	 [['B', 'down', 'right'], ['A', 'B', 'left']] ,
	 [['A', 'B', 'left'], ['NOOP']] ,
	 [['B', 'down', 'left'], ['B', 'down', 'right']] ,
	 [['down', 'right'], ['right']] ,
	 [['B', 'down', 'right'], ['A']] ,
	 [['A', 'B'], ['NOOP']] ,
	 [['B', 'right'], ['A', 'B', 'down']] ,
	 [['NOOP'], ['NOOP']] ,
	 [['A', 'down', 'right'], ['right']] ,
	 [['NOOP'], ['A', 'B', 'down', 'right']] ,
	 [['A', 'left'], ['right']] ,
	 [['down', 'right'], ['A', 'down', 'left']] ,
	 [['A', 'B', 'down', 'right'], ['A', 'down']] ,
	 [['right'], ['B', 'down', 'left']] ,
	 [['down', 'left'], ['B', 'down', 'right']] ,
	 [['A', 'left'], ['down']] ,
	 [['down', 'left'], ['A', 'B', 'left']] ,
	 [['B', 'right'], ['down']] ,
	 [['A', 'right'], ['A', 'right']] ,
	 [['A', 'down'], ['B', 'down', 'left']] ,
	 [['B', 'right'], ['down', 'left']] ,
	 [['B'], ['down', 'right']] ,
	 [['B', 'down'], ['A', 'B', 'down', 'right']] ,
	 [['A', 'B', 'right'], ['A', 'down', 'left']] ,
	 [['left'], ['A', 'down']] ,
	 [['A'], ['A', 'down', 'right']] ,
	 [['down'], ['left']] ,
	 [['B', 'down'], ['A']] ,
	 [['A', 'down', 'left'], ['B', 'down', 'right']] ,
	 [['NOOP'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'B', 'down'], ['A']] ,
	 [['B', 'down'], ['B', 'left']] ,
	 [['left'], ['B', 'right']] ,
	 [['left'], ['A', 'B']] ,
	 [['B', 'down', 'right'], ['B', 'left']] ,
	 [['B', 'down', 'left'], ['A', 'down']] ,
	 [['B', 'right'], ['B', 'down', 'right']] ,
	 [['A', 'down', 'left'], ['B']] ,
	 [['A', 'B', 'down', 'left'], ['A', 'left']] ,
	 [['down', 'left'], ['down']] ,
	 [['A', 'down', 'right'], ['A', 'B', 'down', 'left']] ,
	 [['down'], ['B', 'right']] ,
	 [['down'], ['A', 'down', 'left']] ,
	 [['NOOP'], ['B', 'down', 'left']] ,
	 [['down', 'right'], ['B', 'right']] ,
	 [['A'], ['down', 'right']] ,
	 [['right'], ['A', 'B', 'down']] ,
	 [['A', 'down'], ['A', 'down']] ,
	 [['down', 'left'], ['A', 'left']] ,
	 [['down'], ['NOOP']] ,
	 [['A', 'B', 'down', 'right'], ['down']] ,
	 [['A', 'B', 'right'], ['B', 'down', 'left']] ,
	 [['B', 'down'], ['A', 'down', 'right']] ,
	 [['down', 'right'], ['down', 'left']] ,
	 [['A', 'down', 'left'], ['B', 'down']] ,
	 [['left'], ['A', 'B', 'down']] ,
	 [['B', 'right'], ['A', 'B', 'left']] ,
	 [['A', 'down', 'right'], ['A', 'left']] ,
	 [['A', 'down'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'down'], ['A', 'B', 'down']] ,
	 [['B', 'down', 'right'], ['down', 'left']] ,
	 [['A', 'B', 'left'], ['B']] ,
	 [['A', 'B', 'left'], ['down', 'left']] ,
	 [['A', 'down', 'left'], ['A', 'right']] ,
	 [['B', 'left'], ['A', 'left']] ,
	 [['B'], ['B', 'down']] ,
	 [['NOOP'], ['A', 'left']] ,
	 [['A', 'B', 'down'], ['B', 'right']] ,
	 [['down', 'right'], ['A']] ,
	 [['NOOP'], ['B', 'right']] ,
	 [['B'], ['B', 'right']] ,
	 [['B', 'right'], ['A', 'B']] ,
	 [['A', 'down', 'left'], ['A', 'down', 'right']] ,
	 [['down'], ['A', 'B', 'down', 'left']] ,
	 [['right'], ['A', 'B', 'left']] ,
	 [['A', 'B'], ['A', 'right']] ,
	 [['A', 'B', 'down'], ['A', 'down', 'right']] ,
	 [['A', 'B', 'down', 'left'], ['right']] ,
	 [['B', 'down'], ['left']] ,
	 [['B', 'down'], ['B', 'down', 'left']] ,
	 [['B', 'right'], ['right']] ,
	 [['A', 'down', 'right'], ['down', 'right']] ,
	 [['A', 'down', 'left'], ['A', 'B']] ,
	 [['B', 'down', 'left'], ['NOOP']] ,
	 [['A', 'B', 'down', 'left'], ['A', 'B', 'down']] ,
	 [['right'], ['left']] ,
	 [['A', 'B', 'left'], ['B', 'down']] ,
	 [['B', 'down', 'right'], ['B', 'down', 'right']] ,
	 [['A', 'B', 'down', 'left'], ['A', 'down', 'right']] ,
	 [['A', 'B'], ['right']] ,
	 [['right'], ['A', 'B', 'down', 'left']] ,
	 [['B', 'down', 'left'], ['down', 'left']] ,
	 [['down'], ['A', 'B']] ,
	 [['A', 'B', 'down', 'left'], ['A', 'right']] ,
	 [['A', 'B'], ['B', 'left']] ,
	 [['A', 'down', 'right'], ['A', 'B', 'right']] ,
	 [['A', 'B', 'down'], ['A', 'left']] ,
	 [['B', 'down', 'left'], ['A', 'B', 'right']] ,
	 [['A', 'B', 'down'], ['left']] ,
	 [['A', 'right'], ['A']] ,
	 [['B'], ['A', 'B']] ,
	 [['A', 'left'], ['A']] ,
	 [['B'], ['A', 'B', 'left']] ,
	 [['A', 'B', 'down', 'right'], ['left']] ,
	 [['A', 'B', 'down', 'right'], ['right']] ,
	 [['A', 'B', 'down', 'left'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'B', 'left'], ['B', 'down', 'left']] ,
	 [['A', 'B', 'down', 'left'], ['B', 'left']] ,
	 [['NOOP'], ['A']] ,
	 [['A', 'down', 'right'], ['A', 'down']] ,
	 [['A', 'down'], ['B', 'down']] ,
	 [['left'], ['A', 'down', 'left']] ,
	 [['NOOP'], ['A', 'down', 'right']] ,
	 [['A'], ['A', 'down', 'left']] ,
	 [['A', 'B'], ['A', 'B', 'down', 'right']] ,
	 [['B', 'down', 'right'], ['A', 'left']] ,
	 [['A', 'B', 'right'], ['A', 'B', 'down', 'left']] ,
	 [['B', 'down'], ['B', 'right']] ,
	 [['A', 'B', 'right'], ['down']] ,
	 [['B'], ['A', 'down', 'left']] ,
	 [['down', 'right'], ['NOOP']] ,
	 [['B'], ['down', 'left']] ,
	 [['A', 'left'], ['A', 'B']] ,
	 [['A', 'B', 'left'], ['A', 'B']] ,
	 [['down'], ['A', 'right']] ,
	 [['down', 'left'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'down'], ['A', 'B', 'left']] ,
	 [['A', 'B', 'left'], ['left']] ,
	 [['B', 'left'], ['A', 'B', 'down']] ,
	 [['B', 'down'], ['B', 'down', 'right']] ,
	 [['B', 'left'], ['down', 'right']] ,
	 [['B'], ['A']] ,
	 [['A', 'B'], ['B', 'down']] ,
	 [['B', 'down', 'left'], ['left']] ,
	 [['B', 'down'], ['A', 'B', 'right']] ,
	 [['down'], ['B', 'down']] ,
	 [['B'], ['NOOP']] ,
	 [['A'], ['B', 'down', 'left']] ,
	 [['A', 'down', 'left'], ['A', 'down', 'left']] ,
	 [['B', 'down', 'left'], ['B', 'right']] ,
	 [['B', 'down', 'left'], ['A', 'B', 'down', 'right']] ,
	 [['B', 'down', 'right'], ['left']] ,
	 [['A', 'B', 'down'], ['A', 'down', 'left']] ,
	 [['B', 'right'], ['left']] ,
	 [['down', 'left'], ['down', 'left']] ,
	 [['A', 'right'], ['down', 'right']] ,
	 [['A', 'down', 'right'], ['down', 'left']] ,
	 [['B', 'right'], ['A', 'down']] ,
	 [['B', 'right'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'down'], ['B']] ,
	 [['A', 'right'], ['A', 'B', 'down', 'left']] ,
	 [['B', 'left'], ['A', 'B', 'down', 'right']] ,
	 [['A', 'B'], ['B', 'down', 'left']] ,
	 [['A', 'B', 'down', 'left'], ['A', 'B']] ,
	 [['A', 'B', 'down', 'left'], ['A', 'down', 'left']] ,
	 [['A', 'B', 'left'], ['B', 'left']] ,
	 [['B', 'right'], ['B', 'left']] ,
	 [['down', 'right'], ['A', 'B', 'right']] ,
	 [['right'], ['down', 'right']] ,
	 [['B', 'left'], ['A', 'B', 'left']] ,
	 [['B', 'down', 'left'], ['B', 'down']] ,
	 [['A', 'B', 'down', 'left'], ['down']] ,
	 [['B', 'down'], ['A', 'B', 'down']] ,
	 [['A', 'B', 'down', 'left'], ['A', 'B', 'right']] ,
	 [['A', 'left'], ['A', 'left']] ,
	 [['down', 'right'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'down'], ['A', 'B']] ,
	 [['A', 'right'], ['A', 'B', 'down']] ,
]

TEST_SET = [
	 [['A'], ['A']] ,
	 [['A'], ['A', 'down']] ,
	 [['A'], ['B', 'down', 'right']] ,
	 [['A'], ['B', 'left']] ,
	 [['A'], ['B', 'right']] ,
	 [['A'], ['NOOP']] ,
	 [['A'], ['down']] ,
	 [['A'], ['right']] ,
	 [['A', 'B'], ['A', 'B']] ,
	 [['A', 'B'], ['A', 'down']] ,
	 [['A', 'B'], ['B', 'down', 'right']] ,
	 [['A', 'B'], ['down']] ,
	 [['A', 'B'], ['left']] ,
	 [['A', 'B', 'down'], ['A', 'B', 'down']] ,
	 [['A', 'B', 'down'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'B', 'down'], ['A', 'B', 'left']] ,
	 [['A', 'B', 'down'], ['B', 'down', 'right']] ,
	 [['A', 'B', 'down'], ['B', 'left']] ,
	 [['A', 'B', 'down'], ['down', 'right']] ,
	 [['A', 'B', 'down', 'left'], ['A', 'B', 'down', 'right']] ,
	 [['A', 'B', 'down', 'left'], ['A', 'B', 'left']] ,
	 [['A', 'B', 'down', 'left'], ['A', 'down']] ,
	 [['A', 'B', 'down', 'left'], ['B', 'down']] ,
	 [['A', 'B', 'down', 'left'], ['B', 'down', 'left']] ,
	 [['A', 'B', 'down', 'left'], ['B', 'down', 'right']] ,
	 [['A', 'B', 'down', 'right'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'B', 'down', 'right'], ['A', 'B', 'left']] ,
	 [['A', 'B', 'down', 'right'], ['A', 'left']] ,
	 [['A', 'B', 'left'], ['A', 'B', 'right']] ,
	 [['A', 'B', 'left'], ['A', 'down']] ,
	 [['A', 'B', 'left'], ['A', 'down', 'left']] ,
	 [['A', 'B', 'left'], ['A', 'left']] ,
	 [['A', 'B', 'left'], ['A', 'right']] ,
	 [['A', 'B', 'left'], ['down', 'right']] ,
	 [['A', 'B', 'left'], ['right']] ,
	 [['A', 'B', 'right'], ['A', 'B']] ,
	 [['A', 'B', 'right'], ['A', 'left']] ,
	 [['A', 'B', 'right'], ['B', 'left']] ,
	 [['A', 'B', 'right'], ['NOOP']] ,
	 [['A', 'B', 'right'], ['down', 'left']] ,
	 [['A', 'down'], ['A']] ,
	 [['A', 'down'], ['A', 'down', 'left']] ,
	 [['A', 'down'], ['B', 'down', 'right']] ,
	 [['A', 'down'], ['NOOP']] ,
	 [['A', 'down'], ['down', 'right']] ,
	 [['A', 'down'], ['left']] ,
	 [['A', 'down', 'left'], ['A', 'B', 'down']] ,
	 [['A', 'down', 'left'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'down', 'left'], ['A', 'B', 'down', 'right']] ,
	 [['A', 'down', 'left'], ['A', 'B', 'right']] ,
	 [['A', 'down', 'left'], ['A', 'down']] ,
	 [['A', 'down', 'left'], ['B', 'down', 'left']] ,
	 [['A', 'down', 'left'], ['B', 'left']] ,
	 [['A', 'down', 'left'], ['B', 'right']] ,
	 [['A', 'down', 'left'], ['left']] ,
	 [['A', 'down', 'right'], ['A', 'B', 'left']] ,
	 [['A', 'down', 'right'], ['A', 'down', 'left']] ,
	 [['A', 'down', 'right'], ['B', 'down']] ,
	 [['A', 'down', 'right'], ['B', 'down', 'right']] ,
	 [['A', 'down', 'right'], ['B', 'left']] ,
	 [['A', 'down', 'right'], ['NOOP']] ,
	 [['A', 'down', 'right'], ['down']] ,
	 [['A', 'left'], ['A', 'B', 'down']] ,
	 [['A', 'left'], ['A', 'B', 'right']] ,
	 [['A', 'left'], ['A', 'down']] ,
	 [['A', 'left'], ['B', 'down']] ,
	 [['A', 'left'], ['B', 'down', 'left']] ,
	 [['A', 'left'], ['B', 'down', 'right']] ,
	 [['A', 'left'], ['down', 'left']] ,
	 [['A', 'left'], ['left']] ,
	 [['A', 'right'], ['B']] ,
	 [['A', 'right'], ['B', 'down']] ,
	 [['A', 'right'], ['B', 'down', 'left']] ,
	 [['A', 'right'], ['B', 'left']] ,
	 [['A', 'right'], ['B', 'right']] ,
]

VALIDATION_SET = [
	 [['A', 'right'], ['down', 'left']] ,
	 [['A', 'right'], ['left']] ,
	 [['B'], ['A', 'B', 'down', 'left']] ,
	 [['B'], ['A', 'B', 'right']] ,
	 [['B'], ['B', 'down', 'left']] ,
	 [['B'], ['B', 'down', 'right']] ,
	 [['B'], ['down']] ,
	 [['B', 'down'], ['A', 'B', 'down', 'left']] ,
	 [['B', 'down'], ['A', 'B', 'left']] ,
	 [['B', 'down'], ['A', 'down', 'left']] ,
	 [['B', 'down'], ['B']] ,
	 [['B', 'down'], ['NOOP']] ,
	 [['B', 'down'], ['down']] ,
	 [['B', 'down'], ['down', 'right']] ,
	 [['B', 'down'], ['right']] ,
	 [['B', 'down', 'left'], ['A']] ,
	 [['B', 'down', 'left'], ['A', 'B']] ,
	 [['B', 'down', 'left'], ['A', 'B', 'down']] ,
	 [['B', 'down', 'left'], ['A', 'B', 'left']] ,
	 [['B', 'down', 'left'], ['A', 'down', 'right']] ,
	 [['B', 'down', 'left'], ['down', 'right']] ,
	 [['B', 'down', 'left'], ['right']] ,
	 [['B', 'down', 'right'], ['A', 'B']] ,
	 [['B', 'down', 'right'], ['A', 'B', 'down', 'left']] ,
	 [['B', 'down', 'right'], ['A', 'down', 'left']] ,
	 [['B', 'down', 'right'], ['B']] ,
	 [['B', 'down', 'right'], ['B', 'down', 'left']] ,
	 [['B', 'down', 'right'], ['B', 'right']] ,
	 [['B', 'down', 'right'], ['NOOP']] ,
	 [['B', 'down', 'right'], ['down']] ,
	 [['B', 'left'], ['A', 'B']] ,
	 [['B', 'left'], ['A', 'B', 'down', 'left']] ,
	 [['B', 'left'], ['A', 'B', 'right']] ,
	 [['B', 'left'], ['A', 'down']] ,
	 [['B', 'left'], ['A', 'down', 'right']] ,
	 [['B', 'left'], ['A', 'right']] ,
	 [['B', 'left'], ['B']] ,
	 [['B', 'left'], ['B', 'left']] ,
	 [['B', 'left'], ['left']] ,
	 [['B', 'right'], ['A']] ,
	 [['B', 'right'], ['A', 'down', 'left']] ,
	 [['B', 'right'], ['B', 'right']] ,
	 [['B', 'right'], ['down', 'right']] ,
	 [['NOOP'], ['A', 'B']] ,
	 [['NOOP'], ['A', 'B', 'down']] ,
	 [['NOOP'], ['A', 'B', 'right']] ,
	 [['NOOP'], ['A', 'down', 'left']] ,
	 [['NOOP'], ['B']] ,
	 [['NOOP'], ['B', 'down', 'right']] ,
	 [['NOOP'], ['down', 'left']] ,
	 [['down'], ['A', 'B', 'down', 'right']] ,
	 [['down'], ['A', 'down']] ,
	 [['down'], ['B', 'down', 'left']] ,
	 [['down'], ['B', 'left']] ,
	 [['down'], ['down']] ,
	 [['down'], ['down', 'left']] ,
	 [['down', 'left'], ['A']] ,
	 [['down', 'left'], ['A', 'B']] ,
	 [['down', 'left'], ['A', 'B', 'down', 'right']] ,
	 [['down', 'left'], ['A', 'B', 'right']] ,
	 [['down', 'left'], ['A', 'down', 'left']] ,
	 [['down', 'left'], ['B']] ,
	 [['down', 'left'], ['B', 'down']] ,
	 [['down', 'left'], ['B', 'left']] ,
	 [['down', 'left'], ['NOOP']] ,
	 [['down', 'right'], ['A', 'B', 'down']] ,
	 [['down', 'right'], ['A', 'B', 'left']] ,
	 [['down', 'right'], ['A', 'left']] ,
	 [['down', 'right'], ['B', 'down']] ,
	 [['down', 'right'], ['B', 'left']] ,
	 [['down', 'right'], ['down']] ,
	 [['down', 'right'], ['down', 'right']] ,
	 [['left'], ['A', 'B', 'down', 'right']] ,
	 [['left'], ['A', 'down', 'right']] ,
	 [['left'], ['A', 'right']] ,
	 [['left'], ['B', 'down']] ,
	 [['left'], ['B', 'down', 'left']] ,
	 [['left'], ['B', 'down', 'right']] ,
	 [['left'], ['NOOP']] ,
	 [['left'], ['down', 'right']] ,
	 [['left'], ['right']] ,
	 [['right'], ['A']] ,
	 [['right'], ['down']] ,
]

TRAIN_SET_SUFFICIENT_JUMP_SET =[
    [['A', 'B', 'right'], ['A', 'B', 'right']],
	[['A', 'B', 'right'], ['A', 'B', 'down', 'right']],
	[['A', 'B'], ['A', 'down', 'right']],
	[['A', 'right'], ['A', 'B', 'down', 'right']],
	[['A', 'B', 'down'], ['A', 'down', 'right']],
	[['A', 'right'], ['A', 'down', 'right']],
	[['A', 'right'], ['A', 'right']],
	[['A', 'down', 'right'], ['A', 'B', 'down']],
	[['A', 'B', 'down', 'right'], ['A', 'down']],
	[['A', 'B', 'down', 'right'], ['A', 'B', 'down', 'right']],
	[['A', 'down', 'right'], ['A']],
	[['A', 'right'], ['A', 'B']],
	[['A', 'down', 'right'], ['A', 'B', 'right']],
	[['A', 'B'], ['A', 'B', 'down', 'right']],
	[['A', 'down', 'right'], ['A', 'B', 'down', 'right']],
	[['A'], ['A', 'down', 'right']],
	[['A', 'right'], ['A', 'down']],
	[['A'], ['A', 'B', 'down', 'right']],
	[['A', 'B', 'down', 'right'], ['A', 'down', 'right']],
	[['A', 'down'], ['A', 'down', 'right']],
	[['A', 'B', 'right'], ['A', 'right']],
	[['A', 'down', 'right'], ['A', 'right']],
	[['A', 'right'], ['A']],
	[['A', 'right'], ['A', 'B', 'down']],
	[['A', 'down'], ['A', 'B', 'down', 'right']],
	[['A', 'B', 'right'], ['A', 'down']],
	[['A', 'B', 'right'], ['A', 'down', 'right']],
	[['A'], ['A', 'B', 'right']],
	[['A', 'B', 'down'], ['A', 'right']],
	[['A', 'down', 'right'], ['A', 'down']],
	[['A', 'B'], ['A', 'right']],
	[['A', 'B', 'right'], ['A']],
	[['A', 'B', 'down', 'right'], ['A']]
]

VALIDATION_SET_SUFFICIENT_JUMP_SET = [
    [['A'], ['A', 'right']],
	[['A', 'B', 'down', 'right'], ['A', 'B', 'right']],
	[['A', 'down', 'right'], ['A', 'B']],
	[['A', 'B', 'down'], ['A', 'B', 'down', 'right']],
	[['A', 'B', 'down', 'right'], ['A', 'B', 'down']],
	[['A', 'B', 'down'], ['A', 'B', 'right']],
	[['A', 'B', 'down', 'right'], ['A', 'right']],
	[['A', 'down'], ['A', 'B', 'right']]
]

TEST_SET_SUFFICIENT_JUMP_SET = [
    [['A', 'B', 'right'], ['A', 'B', 'down']],
	[['A', 'B', 'down', 'right'], ['A', 'B']],
	[['A', 'right'], ['A', 'B', 'right']],
	[['A', 'B'], ['A', 'B', 'right']],
	[['A', 'down'], ['A', 'right']],
	[['A', 'down', 'right'], ['A', 'down', 'right']],
	[['A', 'B', 'right'], ['A', 'B']]
]

TRAIN_SET_SUFFICIENT_RIGHT_SET = [
    [['A', 'B', 'down', 'right'], ['B', 'right']],
	[['down', 'right'], ['A', 'down', 'right']],
	[['A', 'down', 'right'], ['B', 'right']],
	[['B', 'down', 'right'], ['A', 'right']],
	[['right'], ['A', 'B', 'right']],
	[['A', 'down', 'right'], ['A', 'B', 'right']],
	[['right'], ['A', 'down', 'right']],
	[['A', 'down', 'right'], ['A', 'B', 'down', 'right']],
	[['A', 'B', 'down', 'right'], ['A', 'B', 'right']],
	[['A', 'B', 'down', 'right'], ['A', 'down', 'right']],
	[['B', 'down', 'right'], ['A', 'down', 'right']],
	[['A', 'down', 'right'], ['A', 'right']],
	[['A', 'B', 'down', 'right'], ['right']],
	[['A', 'down', 'right'], ['down', 'right']],
	[['A', 'down', 'right'], ['A', 'down', 'right']],
	[['A', 'B', 'right'], ['A', 'down', 'right']],
	[['down', 'right'], ['A', 'right']],
	[['B', 'down', 'right'], ['A', 'B', 'right']],
	[['B', 'right'], ['A', 'right']],
	[['B', 'right'], ['A', 'B', 'right']],
	[['A', 'B', 'right'], ['A', 'B', 'right']],
	[['B', 'right'], ['A', 'down', 'right']],
	[['A', 'B', 'right'], ['A', 'right']],
	[['right'], ['A', 'B', 'down', 'right']],
	[['A', 'B', 'down', 'right'], ['A', 'right']],
	[['A', 'down', 'right'], ['right']],
	[['right'], ['A', 'right']],
	[['A', 'right'], ['B', 'down', 'right']],
	[['A', 'right'], ['right']],
	[['A', 'B', 'right'], ['B', 'down', 'right']],
	[['A', 'right'], ['down', 'right']],
	[['A', 'right'], ['A', 'right']]
]

VALIDATION_SET_SUFFICIENT_RIGHT_SET = [
    [['A', 'right'], ['A', 'B', 'down', 'right']],
	[['B', 'down', 'right'], ['A', 'B', 'down', 'right']],
	[['A', 'B', 'right'], ['down', 'right']],
	[['A', 'right'], ['A', 'B', 'right']],
	[['A', 'B', 'down', 'right'], ['A', 'B', 'down', 'right']],
	[['B', 'right'], ['A', 'B', 'down', 'right']],
    [['A', 'B', 'down', 'right'], ['B', 'down', 'right']]
]

TEST_SET_SUFFICIENT_RIGHT_SET = [
    [['A', 'B', 'down', 'right'], ['down', 'right']],
	[['A', 'down', 'right'], ['B', 'down', 'right']],
	[['A', 'B', 'right'], ['right']],
	[['A', 'right'], ['A', 'down', 'right']],
	[['A', 'right'], ['B', 'right']],
	[['down', 'right'], ['A', 'B', 'down', 'right']],
	[['A', 'B', 'right'], ['A', 'B', 'down', 'right']],
]
