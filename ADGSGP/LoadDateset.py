def loadData(flieName):
  inFile = open(flieName, 'r')
  X = []
  y = []
  datasize = int(inFile.readline().split('\t')[0])
  for i in range(0,datasize):
      line=inFile.readline()
      trainingSet = line.split('\t')
      x=[]
      for xi in trainingSet[:-1]:
          x.append(float(xi))
      X.append(tuple(x))
      y.append(float(trainingSet[-1]))
  return X, y