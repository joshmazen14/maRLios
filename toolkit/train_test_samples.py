TRAIN_SET = [
	 [['A', 'down'], ['NOOP']] ,
	 [['B', 'left'], ['down', 'left']] ,
	 [['A', 'left'], ['A', 'down', 'right']] ,
	 [['A'], ['A', 'down', 'right']] ,
	 [['A', 'B', 'down'], ['A']] ,
	 [['B', 'left'], ['B', 'down', 'left']] ,
	 [['B', 'right'], ['down', 'left']] ,
	 [['A', 'B', 'down', 'right'], ['A', 'B', 'left']] ,
	 [['right'], ['A', 'down']] ,
	 [['B', 'down', 'left'], ['NOOP']] ,
	 [['A', 'right'], ['B', 'down', 'left']] ,
	 [['down', 'right'], ['A', 'B', 'down', 'left']] ,
	 [['NOOP'], ['B', 'left']] ,
	 [['A', 'B', 'down'], ['A', 'down', 'right']] ,
	 [['B', 'down', 'left'], ['B']] ,
	 [['B', 'down'], ['A', 'down', 'left']] ,
	 [['A', 'B', 'left'], ['A', 'right']] ,
	 [['A', 'B', 'right'], ['B']] ,
	 [['left'], ['B', 'down', 'left']] ,
	 [['down', 'left'], ['A', 'down', 'left']] ,
	 [['left'], ['B', 'down', 'right']] ,
	 [['A', 'B', 'down', 'right'], ['A', 'down']] ,
	 [['down'], ['B', 'down', 'left']] ,
	 [['A', 'down', 'right'], ['A', 'left']] ,
	 [['B', 'down', 'right'], ['down']] ,
	 [['A', 'left'], ['right']] ,
	 [['B', 'down', 'right'], ['B', 'down']] ,
	 [['A', 'down', 'right'], ['A', 'down', 'left']] ,
	 [['NOOP'], ['A', 'down', 'right']] ,
	 [['down', 'right'], ['A', 'right']] ,
	 [['B'], ['B']] ,
	 [['A', 'B', 'down'], ['A', 'B', 'down', 'right']] ,
	 [['A', 'right'], ['A', 'B', 'left']] ,
	 [['B', 'down', 'right'], ['A', 'B', 'down', 'right']] ,
	 [['down', 'right'], ['B', 'down', 'right']] ,
	 [['B'], ['A']] ,
	 [['B', 'down', 'left'], ['A', 'B', 'right']] ,
	 [['B', 'down', 'left'], ['A', 'B', 'down', 'right']] ,
	 [['A'], ['B', 'down']] ,
	 [['down', 'right'], ['A', 'down', 'right']] ,
	 [['A'], ['A', 'B', 'down', 'left']] ,
	 [['B', 'down'], ['B', 'down']] ,
	 [['A', 'B'], ['A', 'left']] ,
	 [['A', 'B', 'right'], ['A', 'left']] ,
	 [['NOOP'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'B', 'left'], ['A', 'down']] ,
	 [['B', 'down'], ['A', 'B', 'down', 'right']] ,
	 [['A', 'B', 'down'], ['B', 'down', 'right']] ,
	 [['A'], ['down', 'left']] ,
	 [['A', 'down', 'left'], ['NOOP']] ,
	 [['left'], ['A', 'down', 'left']] ,
	 [['A', 'down'], ['down']] ,
	 [['A', 'right'], ['left']] ,
	 [['A', 'B', 'down'], ['A', 'right']] ,
	 [['B', 'down', 'left'], ['A', 'B', 'left']] ,
	 [['A', 'B', 'right'], ['A']] ,
	 [['A', 'down', 'left'], ['A']] ,
	 [['A', 'down'], ['A', 'B', 'down', 'right']] ,
	 [['left'], ['A', 'left']] ,
	 [['A', 'B', 'down', 'right'], ['A', 'B', 'down', 'right']] ,
	 [['A', 'B', 'down', 'left'], ['A', 'down']] ,
	 [['down', 'left'], ['A']] ,
	 [['A', 'down'], ['A', 'B', 'left']] ,
	 [['A', 'down', 'right'], ['A']] ,
	 [['A'], ['A', 'B']] ,
	 [['A', 'down', 'right'], ['down', 'right']] ,
	 [['A', 'B', 'left'], ['A', 'B', 'down']] ,
	 [['B', 'left'], ['A', 'B', 'down']] ,
	 [['B', 'down', 'right'], ['right']] ,
	 [['down', 'left'], ['B', 'down']] ,
	 [['A', 'B', 'left'], ['A', 'B', 'down', 'right']] ,
	 [['left'], ['NOOP']] ,
	 [['A', 'right'], ['A', 'B', 'down']] ,
	 [['B'], ['down', 'right']] ,
	 [['B', 'right'], ['B', 'down', 'left']] ,
	 [['A', 'B', 'down', 'left'], ['B', 'left']] ,
	 [['B', 'down', 'right'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'down', 'right'], ['A', 'B', 'down']] ,
	 [['A', 'left'], ['B', 'down']] ,
	 [['down'], ['A']] ,
	 [['A'], ['B', 'left']] ,
	 [['B'], ['A', 'B', 'down']] ,
	 [['down', 'right'], ['A']] ,
	 [['A', 'B'], ['NOOP']] ,
	 [['A', 'right'], ['A', 'B']] ,
	 [['A', 'B', 'down'], ['A', 'B', 'down']] ,
	 [['right'], ['A', 'B']] ,
	 [['A', 'down', 'left'], ['A', 'left']] ,
	 [['A', 'B', 'down', 'left'], ['B', 'down', 'left']] ,
	 [['A', 'B', 'left'], ['A', 'B', 'right']] ,
	 [['B', 'down', 'left'], ['B', 'down', 'left']] ,
	 [['right'], ['left']] ,
	 [['B', 'right'], ['A', 'B', 'down', 'right']] ,
	 [['B', 'down'], ['A']] ,
	 [['down'], ['down', 'right']] ,
	 [['down', 'left'], ['A', 'B', 'down', 'left']] ,
	 [['B', 'right'], ['A', 'B', 'right']] ,
	 [['down', 'left'], ['down']] ,
	 [['A', 'B', 'down', 'right'], ['A', 'B', 'down']] ,
	 [['B', 'right'], ['A']] ,
	 [['B', 'down', 'right'], ['B']] ,
	 [['B', 'down'], ['B', 'left']] ,
	 [['down', 'left'], ['A', 'down', 'right']] ,
	 [['A', 'down', 'left'], ['B', 'down']] ,
	 [['right'], ['down']] ,
	 [['NOOP'], ['down']] ,
	 [['A', 'B', 'down'], ['B', 'left']] ,
	 [['down'], ['down', 'left']] ,
	 [['A', 'down'], ['B', 'down']] ,
	 [['A', 'B', 'right'], ['A', 'B', 'down', 'left']] ,
	 [['down', 'right'], ['down', 'left']] ,
	 [['B', 'right'], ['NOOP']] ,
	 [['B', 'right'], ['A', 'B', 'down', 'left']] ,
	 [['NOOP'], ['NOOP']] ,
	 [['A', 'B', 'left'], ['right']] ,
	 [['B', 'left'], ['A', 'B', 'left']] ,
	 [['A', 'down', 'left'], ['A', 'B', 'down', 'right']] ,
	 [['A', 'B', 'down'], ['down', 'right']] ,
	 [['A', 'B'], ['down', 'left']] ,
	 [['right'], ['A']] ,
	 [['A', 'down'], ['B', 'down', 'right']] ,
	 [['A', 'B', 'down'], ['B']] ,
	 [['B', 'down', 'right'], ['A']] ,
	 [['A', 'down'], ['left']] ,
	 [['right'], ['NOOP']] ,
	 [['right'], ['A', 'down', 'left']] ,
	 [['A', 'down'], ['right']] ,
	 [['A', 'B'], ['A', 'down']] ,
	 [['B', 'down', 'right'], ['down', 'left']] ,
	 [['B'], ['A', 'B', 'left']] ,
	 [['A', 'B', 'down'], ['B', 'down', 'left']] ,
	 [['A'], ['A', 'right']] ,
	 [['A', 'B', 'down', 'right'], ['B', 'down', 'right']] ,
	 [['A', 'left'], ['B', 'down', 'left']] ,
	 [['A', 'B', 'down', 'left'], ['down', 'left']] ,
	 [['A', 'B', 'down'], ['down', 'left']] ,
	 [['B', 'left'], ['B', 'down', 'right']] ,
	 [['down'], ['A', 'left']] ,
	 [['NOOP'], ['A']] ,
	 [['A', 'left'], ['B', 'right']] ,
	 [['A', 'B', 'down', 'left'], ['A', 'B']] ,
	 [['A', 'B', 'right'], ['A', 'B', 'left']] ,
	 [['A', 'right'], ['B', 'left']] ,
	 [['A', 'right'], ['A', 'down', 'right']] ,
	 [['A', 'B', 'down', 'left'], ['A']] ,
	 [['A', 'B', 'down'], ['A', 'down', 'left']] ,
	 [['A', 'right'], ['B', 'down', 'right']] ,
	 [['right'], ['B', 'right']] ,
	 [['A', 'B', 'right'], ['A', 'B', 'down', 'right']] ,
	 [['A', 'down', 'right'], ['B', 'down', 'left']] ,
	 [['B', 'down'], ['B', 'right']] ,
	 [['right'], ['A', 'B', 'right']] ,
	 [['A', 'B', 'down', 'left'], ['right']] ,
	 [['right'], ['A', 'down', 'right']] ,
	 [['down'], ['A', 'right']] ,
	 [['B', 'right'], ['right']] ,
	 [['B', 'left'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'B', 'down', 'right'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'B'], ['B', 'down', 'right']] ,
	 [['A', 'down', 'left'], ['B']] ,
	 [['B'], ['down', 'left']] ,
	 [['A', 'down'], ['A', 'right']] ,
	 [['A', 'B', 'down'], ['A', 'B', 'right']] ,
	 [['B', 'left'], ['B', 'left']] ,
	 [['down'], ['left']] ,
	 [['A', 'B', 'left'], ['B', 'down']] ,
	 [['A', 'B', 'right'], ['A', 'B', 'right']] ,
	 [['B', 'down'], ['down', 'left']] ,
	 [['A', 'B'], ['A', 'right']] ,
	 [['A', 'B', 'right'], ['B', 'right']] ,
	 [['down', 'right'], ['A', 'B', 'left']] ,
	 [['A', 'B', 'right'], ['A', 'B', 'down']] ,
	 [['NOOP'], ['A', 'B', 'left']] ,
	 [['down', 'left'], ['NOOP']] ,
	 [['A', 'B', 'left'], ['B', 'down', 'right']] ,
	 [['down', 'left'], ['A', 'down']] ,
	 [['A', 'left'], ['down']] ,
	 [['right'], ['B', 'left']] ,
	 [['down', 'right'], ['A', 'down']] ,
	 [['A', 'B'], ['A', 'B', 'down', 'right']] ,
	 [['B', 'down'], ['A', 'down']] ,
	 [['A', 'down', 'left'], ['A', 'down', 'left']] ,
	 [['A', 'B', 'right'], ['down']] ,
	 [['B', 'right'], ['left']] ,
	 [['A', 'right'], ['A', 'right']] ,
	 [['B', 'down', 'right'], ['A', 'B', 'down']] ,
	 [['left'], ['left']] ,
	 [['B', 'left'], ['A', 'right']] ,
	 [['A', 'left'], ['NOOP']] ,
	 [['A', 'down'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'B', 'left'], ['A', 'left']] ,
	 [['A', 'right'], ['A', 'down', 'left']] ,
	 [['down'], ['A', 'down', 'left']] ,
	 [['A'], ['B']] ,
	 [['A', 'B', 'down'], ['B', 'down']] ,
	 [['left'], ['B', 'left']] ,
	 [['down', 'right'], ['B', 'down', 'left']] ,
	 [['right'], ['A', 'right']] ,
	 [['A', 'down', 'right'], ['down', 'left']] ,
	 [['A', 'right'], ['B', 'right']] ,
	 [['B', 'down', 'left'], ['A', 'down', 'left']] ,
	 [['left'], ['A', 'B', 'down']] ,
	 [['B', 'down', 'right'], ['left']] ,
	 [['down'], ['NOOP']] ,
	 [['right'], ['A', 'B', 'down', 'right']] ,
	 [['B', 'down', 'left'], ['A']] ,
	 [['B', 'right'], ['B']] ,
	 [['B', 'left'], ['B']] ,
	 [['A'], ['NOOP']] ,
	 [['A', 'B'], ['A', 'down', 'left']] ,
	 [['B', 'left'], ['A', 'down', 'right']] ,
	 [['A', 'left'], ['A', 'B', 'left']] ,
	 [['B'], ['B', 'down', 'right']] ,
	 [['A', 'down', 'right'], ['right']] ,
	 [['down', 'right'], ['right']] ,
	 [['A', 'down'], ['A', 'B']] ,
	 [['A', 'B', 'down', 'left'], ['A', 'B', 'down', 'right']] ,
	 [['down'], ['down']] ,
	 [['A', 'B', 'left'], ['down']] ,
	 [['B', 'down', 'right'], ['A', 'down', 'left']] ,
	 [['A', 'down', 'right'], ['B', 'down', 'right']] ,
	 [['A', 'left'], ['down', 'left']] ,
	 [['B', 'down', 'right'], ['A', 'B', 'right']] ,
	 [['A', 'B', 'down', 'right'], ['B', 'down', 'left']] ,
	 [['down', 'right'], ['B', 'right']] ,
	 [['B'], ['B', 'down', 'left']] ,
	 [['A', 'down', 'left'], ['A', 'B', 'down']] ,
	 [['A', 'B'], ['right']] ,
	 [['down', 'right'], ['NOOP']] ,
	 [['down', 'left'], ['left']] ,
	 [['A', 'B', 'left'], ['down', 'right']] ,
	 [['B'], ['A', 'right']] ,
	 [['down', 'right'], ['left']] ,
	 [['left'], ['down', 'left']] ,
	 [['B', 'down'], ['right']] ,
	 [['A', 'B', 'down', 'left'], ['A', 'B', 'down', 'left']] ,
	 [['NOOP'], ['down', 'right']] ,
	 [['B'], ['A', 'down', 'right']] ,
	 [['A', 'right'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'B', 'down', 'left'], ['left']] ,
	 [['A', 'down'], ['A', 'down', 'right']] ,
	 [['B', 'left'], ['A', 'B', 'down', 'right']] ,
	 [['A', 'B', 'left'], ['A', 'down', 'right']] ,
	 [['A', 'B'], ['A', 'B', 'left']] ,
	 [['A', 'down', 'right'], ['NOOP']] ,
	 [['A', 'down', 'right'], ['A', 'B', 'right']] ,
	 [['A', 'B', 'left'], ['B']] ,
	 [['A', 'B', 'down', 'right'], ['A', 'down', 'left']] ,
	 [['down', 'right'], ['B', 'down']] ,
	 [['NOOP'], ['right']] ,
	 [['B', 'down'], ['down']] ,
	 [['A', 'B'], ['A', 'B']] ,
	 [['B'], ['right']] ,
	 [['A', 'B', 'down', 'left'], ['A', 'left']] ,
	 [['A', 'B', 'down', 'left'], ['B']] ,
	 [['A', 'B', 'left'], ['A', 'B', 'left']] ,
	 [['B', 'down', 'right'], ['A', 'left']] ,
	 [['A', 'B', 'down', 'right'], ['A']] ,
	 [['B', 'right'], ['down', 'right']] ,
	 [['down'], ['A', 'B', 'left']] ,
	 [['down', 'left'], ['A', 'B', 'left']] ,
	 [['A', 'left'], ['left']] ,
	 [['down', 'right'], ['A', 'B']] ,
	 [['B'], ['A', 'B', 'right']] ,
	 [['A', 'B', 'down', 'left'], ['down', 'right']] ,
	 [['A', 'B'], ['down']] ,
	 [['A', 'B', 'down'], ['left']] ,
	 [['A', 'B'], ['B', 'down', 'left']] ,
	 [['B', 'right'], ['A', 'right']] ,
	 [['B', 'down', 'right'], ['B', 'right']] ,
	 [['A', 'down', 'right'], ['A', 'right']] ,
	 [['A'], ['A', 'B', 'right']] ,
	 [['B', 'down'], ['B']] ,
	 [['left'], ['A', 'B', 'down', 'right']] ,
	 [['down', 'left'], ['B', 'left']] ,
	 [['A', 'down', 'left'], ['A', 'down', 'right']] ,
	 [['A', 'B', 'down'], ['A', 'down']] ,
	 [['A', 'down', 'right'], ['left']] ,
	 [['left'], ['A', 'B', 'right']] ,
	 [['A', 'down', 'left'], ['A', 'B', 'right']] ,
	 [['A', 'B', 'left'], ['down', 'left']] ,
	 [['A', 'down', 'left'], ['A', 'down']] ,
	 [['NOOP'], ['A', 'B', 'down']] ,
	 [['A', 'down', 'right'], ['B']] ,
	 [['B'], ['NOOP']] ,
	 [['A', 'down', 'right'], ['A', 'down', 'right']] ,
	 [['A', 'down', 'left'], ['A', 'right']] ,
	 [['B', 'left'], ['left']] ,
	 [['A', 'down'], ['A', 'down']] ,
	 [['A', 'B', 'down', 'right'], ['A', 'left']] ,
	 [['A'], ['A', 'B', 'left']] ,
	 [['down', 'right'], ['A', 'B', 'down']] ,
	 [['right'], ['right']] ,
	 [['B', 'left'], ['A', 'down']] ,
	 [['B', 'down', 'left'], ['A', 'B', 'down']] ,
	 [['A', 'B', 'down', 'right'], ['A', 'down', 'right']] ,
	 [['down', 'right'], ['B']] ,
	 [['left'], ['A', 'B', 'left']] ,
	 [['A', 'down'], ['B', 'left']] ,
	 [['A', 'B', 'down'], ['A', 'B', 'left']] ,
	 [['B'], ['B', 'down']] ,
	 [['B', 'down', 'left'], ['right']] ,
	 [['NOOP'], ['B', 'down', 'left']] ,
	 [['B', 'down', 'left'], ['B', 'left']] ,
	 [['B', 'down', 'right'], ['B', 'down', 'left']] ,
	 [['B', 'down', 'left'], ['A', 'down']] ,
	 [['down', 'left'], ['down', 'left']] ,
	 [['A', 'B'], ['A', 'B', 'right']] ,
	 [['down'], ['B', 'left']] ,
	 [['NOOP'], ['A', 'B', 'right']] ,
	 [['A', 'B', 'down', 'left'], ['B', 'down']] ,
	 [['A'], ['right']] ,
	 [['A', 'right'], ['A', 'left']] ,
	 [['down'], ['A', 'B', 'down', 'right']] ,
	 [['B', 'down'], ['NOOP']] ,
	 [['A', 'down', 'left'], ['down']] ,
	 [['left'], ['A', 'right']] ,
	 [['NOOP'], ['B']] ,
	 [['A', 'B', 'down', 'right'], ['left']] ,
	 [['A', 'B', 'right'], ['B', 'down', 'right']] ,
	 [['down', 'right'], ['A', 'B', 'right']] ,
	 [['B'], ['A', 'B']] ,
	 [['A', 'down', 'right'], ['down']] ,
	 [['B', 'down', 'left'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'left'], ['A', 'B', 'right']] ,
	 [['down', 'left'], ['B']] ,
	 [['A', 'down'], ['B']] ,
	 [['NOOP'], ['A', 'down', 'left']] ,
	 [['left'], ['B', 'down']] ,
	 [['B', 'down', 'right'], ['A', 'down', 'right']] ,
	 [['down', 'left'], ['A', 'B', 'down']] ,
	 [['down', 'left'], ['B', 'down', 'right']] ,
	 [['A'], ['A', 'B', 'down', 'right']] ,
	 [['A', 'down'], ['A', 'left']] ,
	 [['left'], ['B', 'right']] ,
	 [['A', 'left'], ['A', 'B', 'down', 'right']] ,
	 [['A', 'B', 'right'], ['A', 'down']] ,
	 [['A', 'down', 'right'], ['A', 'B', 'left']] ,
	 [['A', 'down', 'left'], ['left']] ,
	 [['down', 'left'], ['right']] ,
	 [['A', 'B', 'right'], ['NOOP']] ,
	 [['B', 'down', 'right'], ['A', 'B']] ,
	 [['A', 'B', 'right'], ['left']] ,
	 [['A', 'B', 'down', 'left'], ['A', 'B', 'left']] ,
	 [['down'], ['A', 'B', 'down', 'left']] ,
	 [['B', 'right'], ['A', 'B', 'down']] ,
	 [['A', 'B', 'right'], ['A', 'B']] ,
	 [['A', 'left'], ['A', 'B', 'down']] ,
	 [['A', 'down'], ['A', 'down', 'left']] ,
	 [['B', 'right'], ['B', 'down']] ,
	 [['A', 'B', 'left'], ['NOOP']] ,
	 [['A', 'down'], ['B', 'down', 'left']] ,
	 [['B', 'down', 'left'], ['left']] ,
	 [['A', 'down', 'left'], ['A', 'B']] ,
	 [['NOOP'], ['B', 'down', 'right']] ,
	 [['A'], ['down']] ,
	 [['B', 'down', 'left'], ['A', 'B']] ,
	 [['B', 'right'], ['B', 'right']] ,
	 [['A', 'B', 'down'], ['A', 'B']] ,
	 [['A', 'B'], ['B', 'left']] ,
	 [['A', 'right'], ['A', 'down']] ,
	 [['A', 'right'], ['right']] ,
	 [['A', 'B', 'right'], ['A', 'down', 'left']] ,
	 [['B', 'left'], ['NOOP']] ,
	 [['NOOP'], ['B', 'down']] ,
	 [['A', 'B'], ['A', 'down', 'right']] ,
	 [['B', 'down', 'left'], ['down', 'right']] ,
	 [['B', 'left'], ['B', 'down']] ,
	 [['A', 'left'], ['A', 'right']] ,
	 [['A', 'right'], ['B']] ,
	 [['A'], ['A', 'left']] ,
	 [['A', 'B', 'right'], ['A', 'right']] ,
	 [['A', 'B', 'down', 'right'], ['A', 'B', 'right']] ,
	 [['right'], ['A', 'left']] ,
	 [['A', 'down', 'left'], ['right']] ,
	 [['left'], ['A', 'down', 'right']] ,
	 [['B', 'down', 'left'], ['A', 'right']] ,
	 [['A'], ['left']] ,
	 [['down'], ['A', 'B', 'down']] ,
	 [['A', 'down', 'left'], ['B', 'down', 'left']] ,
	 [['down', 'right'], ['A', 'left']] ,
	 [['left'], ['right']] ,
	 [['A', 'B'], ['A']] ,
	 [['A', 'down', 'left'], ['down', 'left']] ,
	 [['A', 'left'], ['A', 'B', 'down', 'left']] ,
	 [['right'], ['A', 'B', 'down']] ,
	 [['A', 'down'], ['A', 'B', 'down']] ,
	 [['NOOP'], ['down', 'left']] ,
	 [['A', 'B', 'left'], ['B', 'left']] ,
	 [['A', 'left'], ['B', 'left']] ,
	 [['A', 'left'], ['A', 'left']] ,
	 [['left'], ['down', 'right']] ,
	 [['down'], ['A', 'down', 'right']] ,
	 [['down'], ['B', 'right']] ,
	 [['right'], ['down', 'left']] ,
	 [['A', 'left'], ['down', 'right']] ,
	 [['B', 'right'], ['B', 'left']] ,
	 [['B', 'down', 'left'], ['B', 'down', 'right']] ,
	 [['B', 'left'], ['B', 'right']] ,
	 [['B', 'down', 'left'], ['down', 'left']] ,
	 [['down'], ['A', 'B']] ,
	 [['A', 'right'], ['B', 'down']] ,
	 [['left'], ['A', 'down']]
]

TEST_SET = [
	 [['A'], ['A']] ,
	 [['A'], ['A', 'B', 'down']] ,
	 [['A'], ['A', 'down']] ,
	 [['A'], ['A', 'down', 'left']] ,
	 [['A'], ['B', 'down', 'left']] ,
	 [['A'], ['B', 'down', 'right']] ,
	 [['A'], ['B', 'right']] ,
	 [['A'], ['down', 'right']] ,
	 [['A', 'B'], ['A', 'B', 'down']] ,
	 [['A', 'B'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'B'], ['B']] ,
	 [['A', 'B'], ['B', 'down']] ,
	 [['A', 'B'], ['B', 'right']] ,
	 [['A', 'B'], ['down', 'right']] ,
	 [['A', 'B'], ['left']] ,
	 [['A', 'B', 'down'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'B', 'down'], ['A', 'left']] ,
	 [['A', 'B', 'down'], ['B', 'right']] ,
	 [['A', 'B', 'down'], ['NOOP']] ,
	 [['A', 'B', 'down'], ['down']] ,
	 [['A', 'B', 'down'], ['right']] ,
	 [['A', 'B', 'down', 'left'], ['A', 'B', 'down']] ,
	 [['A', 'B', 'down', 'left'], ['A', 'B', 'right']] ,
	 [['A', 'B', 'down', 'left'], ['A', 'down', 'left']] ,
	 [['A', 'B', 'down', 'left'], ['A', 'down', 'right']] ,
	 [['A', 'B', 'down', 'left'], ['A', 'right']] ,
	 [['A', 'B', 'down', 'left'], ['B', 'down', 'right']] ,
	 [['A', 'B', 'down', 'left'], ['B', 'right']] ,
	 [['A', 'B', 'down', 'left'], ['NOOP']] ,
	 [['A', 'B', 'down', 'left'], ['down']] ,
	 [['A', 'B', 'down', 'right'], ['A', 'B']] ,
	 [['A', 'B', 'down', 'right'], ['A', 'right']] ,
	 [['A', 'B', 'down', 'right'], ['B']] ,
	 [['A', 'B', 'down', 'right'], ['B', 'down']] ,
	 [['A', 'B', 'down', 'right'], ['B', 'left']] ,
	 [['A', 'B', 'down', 'right'], ['B', 'right']] ,
	 [['A', 'B', 'down', 'right'], ['NOOP']] ,
	 [['A', 'B', 'down', 'right'], ['down']] ,
	 [['A', 'B', 'down', 'right'], ['down', 'left']] ,
	 [['A', 'B', 'down', 'right'], ['down', 'right']] ,
	 [['A', 'B', 'down', 'right'], ['right']] ,
	 [['A', 'B', 'left'], ['A']] ,
	 [['A', 'B', 'left'], ['A', 'B']] ,
	 [['A', 'B', 'left'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'B', 'left'], ['A', 'down', 'left']] ,
	 [['A', 'B', 'left'], ['B', 'down', 'left']] ,
	 [['A', 'B', 'left'], ['B', 'right']] ,
	 [['A', 'B', 'left'], ['left']] ,
	 [['A', 'B', 'right'], ['A', 'down', 'right']] ,
	 [['A', 'B', 'right'], ['B', 'down']] ,
	 [['A', 'B', 'right'], ['B', 'down', 'left']] ,
	 [['A', 'B', 'right'], ['B', 'left']] ,
	 [['A', 'B', 'right'], ['down', 'left']] ,
	 [['A', 'B', 'right'], ['down', 'right']] ,
	 [['A', 'B', 'right'], ['right']] ,
	 [['A', 'down'], ['A']] ,
	 [['A', 'down'], ['A', 'B', 'right']] ,
	 [['A', 'down'], ['B', 'right']] ,
	 [['A', 'down'], ['down', 'left']] ,
	 [['A', 'down'], ['down', 'right']] ,
	 [['A', 'down', 'left'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'down', 'left'], ['A', 'B', 'left']] ,
	 [['A', 'down', 'left'], ['B', 'down', 'right']] ,
	 [['A', 'down', 'left'], ['B', 'left']] ,
	 [['A', 'down', 'left'], ['B', 'right']] ,
	 [['A', 'down', 'left'], ['down', 'right']] ,
	 [['A', 'down', 'right'], ['A', 'B']] ,
	 [['A', 'down', 'right'], ['A', 'B', 'down', 'left']] ,
	 [['A', 'down', 'right'], ['A', 'B', 'down', 'right']] ,
	 [['A', 'down', 'right'], ['A', 'down']] ,
	 [['A', 'down', 'right'], ['B', 'down']] ,
	 [['A', 'down', 'right'], ['B', 'left']] ,
	 [['A', 'down', 'right'], ['B', 'right']] ,
	 [['A', 'left'], ['A']] ,
	 [['A', 'left'], ['A', 'B']] ,
	 [['A', 'left'], ['A', 'down']] ,
	 [['A', 'left'], ['A', 'down', 'left']] ,
	 [['A', 'left'], ['B']] ,
	 [['A', 'left'], ['B', 'down', 'right']] ,
	 [['A', 'right'], ['A']] ,
	 [['A', 'right'], ['A', 'B', 'down', 'right']] ,
	 [['A', 'right'], ['A', 'B', 'right']] ,
	 [['A', 'right'], ['NOOP']] ,
	 [['A', 'right'], ['down']] ,
	 [['A', 'right'], ['down', 'left']] ,
	 [['A', 'right'], ['down', 'right']] ,
	 [['B'], ['A', 'B', 'down', 'left']] ,
	 [['B'], ['A', 'B', 'down', 'right']] ,
	 [['B'], ['A', 'down']] ,
	 [['B'], ['A', 'down', 'left']] ,
	 [['B'], ['A', 'left']] ,
	 [['B'], ['B', 'left']] ,
	 [['B'], ['B', 'right']] ,
	 [['B'], ['down']] ,
	 [['B'], ['left']] ,
	 [['B', 'down'], ['A', 'B']] ,
	 [['B', 'down'], ['A', 'B', 'down']] ,
	 [['B', 'down'], ['A', 'B', 'down', 'left']] ,
	 [['B', 'down'], ['A', 'B', 'left']] ,
	 [['B', 'down'], ['A', 'B', 'right']] ,
	 [['B', 'down'], ['A', 'down', 'right']] ,
	 [['B', 'down'], ['A', 'left']] ,
	 [['B', 'down'], ['A', 'right']] ,
	 [['B', 'down'], ['B', 'down', 'left']] ,
	 [['B', 'down'], ['B', 'down', 'right']] ,
	 [['B', 'down'], ['down', 'right']] ,
	 [['B', 'down'], ['left']] ,
	 [['B', 'down', 'left'], ['A', 'down', 'right']] ,
	 [['B', 'down', 'left'], ['A', 'left']] ,
	 [['B', 'down', 'left'], ['B', 'down']] ,
	 [['B', 'down', 'left'], ['B', 'right']] ,
	 [['B', 'down', 'left'], ['down']] ,
	 [['B', 'down', 'right'], ['A', 'B', 'left']] ,
	 [['B', 'down', 'right'], ['A', 'down']] ,
	 [['B', 'down', 'right'], ['A', 'right']] ,
	 [['B', 'down', 'right'], ['B', 'down', 'right']] ,
	 [['B', 'down', 'right'], ['B', 'left']] ,
	 [['B', 'down', 'right'], ['NOOP']] ,
	 [['B', 'down', 'right'], ['down', 'right']] ,
	 [['B', 'left'], ['A']] ,
	 [['B', 'left'], ['A', 'B']] ,
	 [['B', 'left'], ['A', 'B', 'right']] ,
	 [['B', 'left'], ['A', 'down', 'left']] ,
	 [['B', 'left'], ['A', 'left']] ,
	 [['B', 'left'], ['down']] ,
	 [['B', 'left'], ['down', 'right']] ,
	 [['B', 'left'], ['right']] ,
	 [['B', 'right'], ['A', 'B']] ,
	 [['B', 'right'], ['A', 'B', 'left']] ,
	 [['B', 'right'], ['A', 'down']] ,
	 [['B', 'right'], ['A', 'down', 'left']] ,
	 [['B', 'right'], ['A', 'down', 'right']] ,
	 [['B', 'right'], ['A', 'left']] ,
	 [['B', 'right'], ['B', 'down', 'right']] ,
	 [['B', 'right'], ['down']] ,
	 [['NOOP'], ['A', 'B']] ,
	 [['NOOP'], ['A', 'B', 'down', 'right']] ,
	 [['NOOP'], ['A', 'down']] ,
	 [['NOOP'], ['A', 'left']] ,
	 [['NOOP'], ['A', 'right']] ,
	 [['NOOP'], ['B', 'right']] ,
	 [['NOOP'], ['left']] ,
	 [['down'], ['A', 'B', 'right']] ,
	 [['down'], ['A', 'down']] ,
	 [['down'], ['B']] ,
	 [['down'], ['B', 'down']] ,
	 [['down'], ['B', 'down', 'right']] ,
	 [['down'], ['right']] ,
	 [['down', 'left'], ['A', 'B']] ,
	 [['down', 'left'], ['A', 'B', 'down', 'right']] ,
	 [['down', 'left'], ['A', 'B', 'right']] ,
	 [['down', 'left'], ['A', 'left']] ,
	 [['down', 'left'], ['A', 'right']] ,
	 [['down', 'left'], ['B', 'down', 'left']] ,
	 [['down', 'left'], ['B', 'right']] ,
	 [['down', 'left'], ['down', 'right']] ,
	 [['down', 'right'], ['A', 'B', 'down', 'right']] ,
	 [['down', 'right'], ['A', 'down', 'left']] ,
	 [['down', 'right'], ['B', 'left']] ,
	 [['down', 'right'], ['down']] ,
	 [['down', 'right'], ['down', 'right']] ,
	 [['left'], ['A']] ,
	 [['left'], ['A', 'B']] ,
	 [['left'], ['A', 'B', 'down', 'left']] ,
	 [['left'], ['B']] ,
	 [['left'], ['down']] ,
	 [['right'], ['A', 'B', 'down', 'left']] ,
	 [['right'], ['A', 'B', 'left']] ,
	 [['right'], ['B']] ,
	 [['right'], ['B', 'down']] ,
	 [['right'], ['B', 'down', 'left']] ,
	 [['right'], ['B', 'down', 'right']] ,
	 [['right'], ['down', 'right']] 
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
